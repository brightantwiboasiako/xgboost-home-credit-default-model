"""Build a model-ready feature matrix from the 10 Home Credit CSV tables.

THE CORE PROBLEM
----------------
We must predict, for each loan applicant, whether they will default. The
applicant is identified by SK_ID_CURR. XGBoost needs a rectangular matrix with
exactly ONE ROW PER APPLICANT, but only `application_train.csv` is shaped that
way. The other six tables have MANY rows per applicant:

    application_train    1 row  per applicant          <- target lives here
    bureau               N rows per applicant          (one per external loan)
    bureau_balance       N rows per external loan      (one per month)
    previous_application N rows per applicant          (one per past application)
    POS_CASH_balance     N rows per past application   (one per month)
    installments_payments N rows per past application  (one per payment made)
    credit_card_balance  N rows per past application   (one per month)

So every auxiliary table follows the SAME THREE STEPS in this module:

    1. one-hot encode its categorical columns (so they can be averaged)
    2. groupby(SK_ID_CURR).agg([...])  -> collapses to one row per applicant
    3. left-join the result onto the application frame

This "aggregate then join" pattern is the whole competition. The application
table alone scores ~0.74 AUC; adding these aggregates gets to ~0.78.

WHY AGGREGATION CAPTURES SIGNAL
-------------------------------
A single applicant might have 30 installment payments. Any one payment tells us
little, but `mean(days_late)` tells us whether they habitually pay late, and
`max(days_late)` tells us their worst-ever delinquency. Those summary statistics
are what the model can actually learn from.

WHY WE NEVER FILL NaN
---------------------
Left-joins produce NaN for applicants absent from a table (e.g. someone with no
credit history at all). We deliberately leave those NaN. XGBoost learns a
default branch direction for missing values at every split, and "this person has
no bureau record" is itself a meaningful risk signal that imputation would erase.
"""

import gc
import numpy as np
import pandas as pd

# The five summary statistics applied to most numeric columns.
#   mean -> typical behaviour        max/min -> best and worst case
#   sum  -> total exposure           std     -> volatility / consistency
# Each source column therefore expands into five features.
NUM_AGGS = ["mean", "max", "min", "sum", "std"]

# Time windows in MONTHS for the recency-weighted aggregates. A customer's
# lifetime mean hides the thing that matters most -- someone who paid perfectly
# for four years and started missing payments last quarter looks identical to a
# steady payer once you average over all of it. Computing the same statistics
# over shrinking windows lets the model see recent behaviour separately, and the
# deltas between windows (see _window_aggs) expose the direction of travel.
WINDOWS = [3, 6, 12, 24]

# Median rather than mean inside the windows. Payment tables carry heavy-tailed
# outliers -- a single settlement payment or one 300-day delinquency drags a
# mean far from typical behaviour, especially in a 3-month window holding only a
# handful of rows. The median describes what the customer usually does, which is
# what the trend and delta features are meant to compare.
WINDOW_STAT = "median"


def _months(series, unit):
    """Convert a time-offset column to months before the application date.

    Source tables disagree: installments_payments and bureau count DAYS
    (negative, to about -2922), while credit_card_balance and POS_CASH_balance
    count MONTHS (negative, to about -96). Both become positive month counts so
    one windowing rule works everywhere.
    """
    return -series / 30.0 if unit == "days" else -series


def _window_aggs(df, key, time_col, value_cols, prefix, unit="days"):
    """Median of each value column within trailing time windows, plus deltas.

    For every window in WINDOWS this emits <PREFIX>_<COL>_<STAT>_<N>M, and for
    every pair of adjacent windows a delta column <PREFIX>_<COL>_DELTA_3_12M
    style. The deltas are the point: a raw window median says what someone did
    recently, but recent-minus-older says whether they are getting better or
    worse, which is the actual risk signal.

    Rows outside a window are absent rather than zero-filled, so a customer with
    no activity in the last 3 months yields NaN there -- correct, since "no
    recent payments" is not the same as "recent payments of zero".
    """
    months = _months(df[time_col], unit)
    out = {}

    for w in WINDOWS:
        sub = df[months <= w]
        if sub.empty:
            continue
        agg = sub.groupby(key)[value_cols].agg(WINDOW_STAT)
        agg.columns = [f"{prefix}_{c}_{WINDOW_STAT.upper()}_{w}M"
                       for c in agg.columns]
        out[w] = agg

    if not out:
        return pd.DataFrame()

    result = pd.concat(out.values(), axis=1)

    # Deltas between adjacent windows: recent behaviour minus older behaviour.
    # Positive on a lateness column means the customer is slipping.
    avail = sorted(out.keys())
    deltas = {}
    for short, long in zip(avail, avail[1:]):
        for c in value_cols:
            s = f"{prefix}_{c}_{WINDOW_STAT.upper()}_{short}M"
            l = f"{prefix}_{c}_{WINDOW_STAT.upper()}_{long}M"
            if s in result.columns and l in result.columns:
                deltas[f"{prefix}_{c}_DELTA_{short}_{long}M"] = (
                    result[s] - result[l])

    if deltas:
        result = pd.concat([result, pd.DataFrame(deltas, index=result.index)],
                           axis=1)
    return result


def _trend(df, key, time_col, value_cols, prefix, unit="days", min_points=3):
    """Least-squares slope of each value column against time, per customer.

    Answers "is this deteriorating?" in a single number per column. The slope is
    computed against months-before-application negated, so time increases toward
    the present and a POSITIVE slope means the value is RISING over time -- on a
    lateness or utilisation column, that is the customer getting worse.

    Implemented as a closed-form covariance/variance ratio rather than
    np.polyfit per group: polyfit on hundreds of thousands of groups is
    prohibitively slow, whereas this is a handful of vectorised groupby passes.

    Customers with fewer than `min_points` observations get NaN -- a slope
    through one or two points is noise, not a trend.
    """
    t = -_months(df[time_col], unit)  # increasing toward the present
    work = pd.DataFrame({key: df[key].values, "_t": t.values})
    for c in value_cols:
        work[c] = df[c].values

    g = work.groupby(key)
    n = g["_t"].size()
    t_mean = g["_t"].mean()

    # var(t) is the denominator of the slope; zero when every observation shares
    # one timestamp, which would divide by zero.
    t_var = g["_t"].apply(lambda s: ((s - s.mean()) ** 2).sum())

    out = {}
    for c in value_cols:
        # cov(t, y) computed as sum((t - t_bar) * y); the (y - y_bar) term
        # cancels in the sum, so this is exact and cheaper.
        work["_ty"] = (work["_t"] - work[key].map(t_mean)) * work[c]
        cov = work.groupby(key)["_ty"].sum()
        slope = cov / t_var.replace(0, np.nan)
        slope[n < min_points] = np.nan
        out[f"{prefix}_{c}_TREND"] = slope

    return pd.DataFrame(out)


def _one_hot(df):
    """Convert every text column into 0/1 indicator columns.

    Why this is needed: groupby().mean() cannot average a string. After one-hot
    encoding, the mean of an indicator column has a genuinely useful reading --
    e.g. mean of CREDIT_ACTIVE_Active becomes "what fraction of this applicant's
    external loans are still active", which is exactly the kind of ratio feature
    we want.

    dummy_na=True adds an explicit column for missing categories, because
    "category was not recorded" can itself be predictive.

    Returns:
        (dataframe, list_of_new_dummy_column_names)

        Returning ONLY the newly created names matters. An earlier version
        returned every column here, which made the caller's "numeric columns"
        filter exclude everything and crash the aggregation with
        "No objects to concatenate".
    """
    # is_string_dtype rather than `dtype == "object"`: pandas 3.0 reads text
    # columns as the Arrow-backed "str" dtype, so the object comparison silently
    # matches nothing and the strings survive into groupby().mean(), which fails
    # with "dtype 'str' does not support operation 'mean'". This check is correct
    # on both pandas 2.x and 3.x.
    cats = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
    before = set(df.columns)
    df = pd.get_dummies(df, columns=cats, dummy_na=True)
    new = [c for c in df.columns if c not in before]

    # pandas makes dummies bool; cast to uint8 so that groupby sum/mean do
    # arithmetic rather than logical operations, and to halve memory use.
    # Cast as one block: a per-column loop would rewrite the frame once per
    # dummy, and get_dummies can produce hundreds of them.
    if new:
        df = df.astype({c: "uint8" for c in new})
    return df, new


def _flatten(df, prefix):
    """Collapse the two-level column index left behind by .agg().

    groupby().agg({"AMT_CREDIT": ["mean", "max"]}) produces a MultiIndex like
    ("AMT_CREDIT", "mean"). XGBoost needs flat unique strings, and we also need
    to know which source table each feature came from once ~1,600 of them are
    sitting in one matrix. So ("AMT_CREDIT", "mean") becomes BURO_AMT_CREDIT_MEAN.
    """
    df.columns = pd.Index([f"{prefix}_{a}_{b}".upper() for a, b in df.columns])
    return df


def application(data_dir, nrows=None):
    """Load the main application table and engineer hand-built ratio features.

    train and test are concatenated before encoding. This is important: if each
    were one-hot encoded separately, a category present in train but absent in
    test would produce different column sets and the matrices would not align.

    Returns:
        (combined_dataframe, number_of_training_rows)
    """
    train = pd.read_csv(f"{data_dir}/application_train.csv", nrows=nrows)
    test = pd.read_csv(f"{data_dir}/application_test.csv", nrows=nrows)
    n_train = len(train)

    # Test rows simply have no TARGET column value; that NaN is how we split
    # them apart again at the end of build().
    df = pd.concat([train, test], ignore_index=True, sort=False)

    # Four rows record gender as "XNA", which is a missing-data code rather
    # than a real category. Dropping them avoids a junk indicator column.
    df = df[df["CODE_GENDER"] != "XNA"]

    # ---- Sentinel value cleanup -------------------------------------------
    # Home Credit encodes "not applicable" as 365243 in day-offset columns.
    # Those columns are negative day counts relative to the application date,
    # so a raw 365243 reads as roughly +1000 YEARS. Left in place it is a
    # massive outlier that drags split thresholds away from the real data.
    # NaN is the honest encoding and XGBoost handles it natively.
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Derived columns are assembled in a dict and attached in ONE concat below.
    # Assigning them one at a time with df["NEW"] = ... inserts into a frame
    # that already has ~270 columns, and each insert copies the whole block
    # manager -- which is what triggers pandas' "DataFrame is highly fragmented"
    # PerformanceWarning. One concat is a single allocation instead of nine.
    ext = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    derived = {
        # ---- Affordability ratios ------------------------------------------
        # Raw amounts are weak predictors because the same loan means very
        # different things at different income levels. A $50k loan is routine on
        # a $200k income and alarming on a $20k income. Ratios encode that
        # directly, saving the model from discovering it through splits.
        "CREDIT_INCOME_RATIO": df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"],
        "ANNUITY_INCOME_RATIO": df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"],

        # Annuity / credit approximates the repayment rate, i.e. loan term.
        "CREDIT_TERM": df["AMT_ANNUITY"] / df["AMT_CREDIT"],

        # Both are negative day counts, so this is the fraction of the
        # applicant's life spent in their current job -- employment stability.
        "EMPLOYED_BIRTH_RATIO": df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"],

        # Household income per head, rather than raw household income.
        "INCOME_PER_PERSON": df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"],

        # Credit far above the price of the goods suggests cash extracted
        # beyond the purchase itself, which historically carries more risk.
        "GOODS_CREDIT_RATIO": df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"],

        # ---- External credit scores ----------------------------------------
        # EXT_SOURCE_1/2/3 are normalised scores from outside credit agencies
        # and are consistently the strongest features in this competition.
        # Combining them adds information beyond the three raw columns: the mean
        # is a more stable consensus, the std flags applicants the agencies
        # disagree about, and the product is a sharp non-linear interaction.
        "EXT_SOURCE_MEAN": df[ext].mean(axis=1),
        "EXT_SOURCE_STD": df[ext].std(axis=1),
        "EXT_SOURCE_PROD": df[ext[0]] * df[ext[1]] * df[ext[2]],
    }
    df = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)

    df, _ = _one_hot(df)
    return df, n_train


def bureau(data_dir, nrows=None):
    """Aggregate credit-bureau records: loans held with OTHER lenders.

    This is a two-level aggregation, the only one in the pipeline:

        bureau_balance (monthly rows, keyed by SK_ID_BUREAU)
            -> aggregate to one row per external loan
        bureau (one row per external loan, keyed by SK_ID_CURR)
            -> aggregate to one row per applicant

    The two-step is required because bureau_balance has no SK_ID_CURR column;
    it can only reach the applicant by joining through bureau first.
    """
    bur = pd.read_csv(f"{data_dir}/bureau.csv", nrows=nrows)
    bal = pd.read_csv(f"{data_dir}/bureau_balance.csv", nrows=nrows)

    # --- Level 1: monthly balance rows -> one row per external loan ---------
    bal, bal_dummies = _one_hot(bal)
    bal_agg = bal.groupby("SK_ID_BUREAU").agg({
        # min/max give the observed history window; size is its length in months.
        "MONTHS_BALANCE": ["min", "max", "size"],
        # Mean of each STATUS indicator = fraction of months in that status,
        # e.g. the share of months the loan was in arrears.
        **{c: ["mean"] for c in bal_dummies},
    })
    bal_agg = _flatten(bal_agg, "BB")

    bur = bur.join(bal_agg, how="left", on="SK_ID_BUREAU")
    bur.drop(columns=["SK_ID_BUREAU"], inplace=True)
    del bal, bal_agg
    gc.collect()  # these frames are large; free them before the next step

    # --- Level 2: external loans -> one row per applicant -------------------
    bur, _ = _one_hot(bur)
    num_cols = [c for c in bur.columns if c != "SK_ID_CURR"]
    agg = bur.groupby("SK_ID_CURR").agg({c: NUM_AGGS for c in num_cols})
    agg = _flatten(agg, "BURO")

    # Sheer number of external loans is predictive on its own. Built as a frame
    # so it joins in one allocation rather than fragmenting `agg`.
    parts = [agg, bur.groupby("SK_ID_CURR").size().to_frame("BURO_COUNT")]

    # Currently-active debt elsewhere is a sharper risk signal than debt that
    # has already been repaid, so recompute key aggregates on active loans only.
    if "CREDIT_ACTIVE_Active" in bur.columns:
        act = bur[bur["CREDIT_ACTIVE_Active"] == 1]
        act_agg = act.groupby("SK_ID_CURR").agg({
            c: ["mean", "sum"]
            for c in ("AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "DAYS_CREDIT")
            if c in act.columns
        })
        # Reindexed to agg's applicants so concat aligns without reordering;
        # applicants with no active loans correctly become NaN.
        parts.append(_flatten(act_agg, "BURO_ACT").reindex(agg.index))

    agg = pd.concat(parts, axis=1)

    del bur
    gc.collect()
    return agg


def previous_application(data_dir, nrows=None):
    """Aggregate the applicant's earlier applications to Home Credit itself.

    Unlike `bureau` (other lenders), this is Home Credit's own history with the
    applicant, including applications that were REFUSED -- which turns out to be
    the most valuable part of the table.
    """
    prev = pd.read_csv(f"{data_dir}/previous_application.csv", nrows=nrows)

    # Same 365243 "not applicable" sentinel as in application(), here spread
    # across five different day-offset columns.
    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
              "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        prev[c] = prev[c].replace(365243, np.nan)

    # How much was asked for vs how much was granted. A ratio well above 1
    # means the applicant was repeatedly offered less than they requested.
    #
    # GUARD: AMT_CREDIT is 0 on ~337k cancelled applications, so this divide
    # yields inf. XGBoost rejects inf outright ("Input data contains `inf`"),
    # and this crashed the first full training run. NaN is both accepted and
    # semantically correct -- the ratio is undefined, not infinite.
    prev["APP_CREDIT_RATIO"] = (prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
                                ).replace([np.inf, -np.inf], np.nan)

    prev, _ = _one_hot(prev)
    num_cols = [c for c in prev.columns if c not in ("SK_ID_PREV", "SK_ID_CURR")]
    agg = prev.groupby("SK_ID_CURR").agg({c: NUM_AGGS for c in num_cols})
    agg = _flatten(agg, "PREV")

    # Attached together in one concat -- `agg` is already ~900 columns wide, so
    # appending them individually fragments the frame (see application()).
    extra = {"PREV_COUNT": prev.groupby("SK_ID_CURR").size()}

    # Fraction of past applications that were refused. Home Credit's own prior
    # rejections encode risk assessments the lender already made about this
    # person, so this is among the most informative engineered features here.
    if "NAME_CONTRACT_STATUS_Refused" in prev.columns:
        extra["PREV_REFUSED_RATE"] = prev.groupby(
            "SK_ID_CURR")["NAME_CONTRACT_STATUS_Refused"].mean()

    agg = pd.concat([agg, pd.DataFrame(extra)], axis=1)

    del prev
    gc.collect()
    return agg


def pos_cash(data_dir, nrows=None):
    """Aggregate monthly point-of-sale / cash-loan balance snapshots.

    Only four statistics here rather than the full NUM_AGGS set: this table is
    ~10M rows and mostly tracks remaining instalment counts, where sum and std
    add little over mean/max/min while costing memory and training time.
    """
    pos = pd.read_csv(f"{data_dir}/POS_CASH_balance.csv", nrows=nrows)
    pos, _ = _one_hot(pos)
    num_cols = [c for c in pos.columns if c not in ("SK_ID_PREV", "SK_ID_CURR")]
    agg = pos.groupby("SK_ID_CURR").agg(
        {c: ["mean", "max", "min", "size"] for c in num_cols})
    agg = _flatten(agg, "POS")

    # SK_DPD tracks days past due on the instalment loan; its trend shows
    # whether arrears are building.
    behaviour = [c for c in ("SK_DPD", "SK_DPD_DEF", "CNT_INSTALMENT_FUTURE")
                 if c in pos.columns]
    agg = pd.concat([
        agg,
        _window_aggs(pos, "SK_ID_CURR", "MONTHS_BALANCE", behaviour,
                     "POS", unit="months").reindex(agg.index),
        _trend(pos, "SK_ID_CURR", "MONTHS_BALANCE", behaviour,
               "POS", unit="months").reindex(agg.index),
    ], axis=1)

    del pos
    gc.collect()
    return agg


def installments(data_dir, nrows=None):
    """Aggregate individual repayments — the single most informative table.

    Everything else describes the applicant's situation; this table records
    their actual repayment BEHAVIOUR. The four derived columns below are the
    entire point of loading it: raw payment amounts matter far less than the
    gap between what was owed and what was actually paid, and when.
    """
    ins = pd.read_csv(f"{data_dir}/installments_payments.csv", nrows=nrows)

    # Fraction of the instalment actually paid. Below 1 means underpayment.
    #
    # GUARD: 290 rows have AMT_INSTALMENT == 0, producing inf. Same crash and
    # same reasoning as APP_CREDIT_RATIO above.
    ins["PAYMENT_PERC"] = (ins["AMT_PAYMENT"] / ins["AMT_INSTALMENT"]
                           ).replace([np.inf, -np.inf], np.nan)

    # Absolute shortfall in currency terms; positive means underpaid.
    ins["PAYMENT_DIFF"] = ins["AMT_INSTALMENT"] - ins["AMT_PAYMENT"]

    # Both columns are negative day offsets, so entry minus due is positive
    # when the payment landed after the due date.
    #   DPD = days past due   (lateness; clipped so early payments read as 0)
    #   DBD = days before due (earliness; clipped so late payments read as 0)
    # Splitting one signed quantity into two clipped columns lets the model
    # treat lateness and earliness asymmetrically, which matches reality:
    # being 30 days late matters far more than being 30 days early.
    ins["DPD"] = (ins["DAYS_ENTRY_PAYMENT"] - ins["DAYS_INSTALMENT"]).clip(lower=0)
    ins["DBD"] = (ins["DAYS_INSTALMENT"] - ins["DAYS_ENTRY_PAYMENT"]).clip(lower=0)

    num_cols = [c for c in ins.columns if c not in ("SK_ID_PREV", "SK_ID_CURR")]
    agg = ins.groupby("SK_ID_CURR").agg({c: NUM_AGGS for c in num_cols})
    agg = _flatten(agg, "INS")

    # Total number of payments made, i.e. depth of repayment history.
    parts = [agg, ins.groupby("SK_ID_CURR").size().to_frame("INS_COUNT")]

    # Recency and trend on the four behavioural columns. This is the table where
    # deterioration shows up first: people stop paying on time before they
    # default outright, so recent lateness carries far more signal than the
    # lifetime average that the aggregates above capture.
    behaviour = ["DPD", "DBD", "PAYMENT_PERC", "PAYMENT_DIFF"]
    parts.append(_window_aggs(ins, "SK_ID_CURR", "DAYS_INSTALMENT", behaviour,
                              "INS", unit="days").reindex(agg.index))
    parts.append(_trend(ins, "SK_ID_CURR", "DAYS_INSTALMENT", behaviour,
                        "INS", unit="days").reindex(agg.index))

    agg = pd.concat(parts, axis=1)
    del ins
    gc.collect()
    return agg


def credit_card(data_dir, nrows=None):
    """Aggregate monthly credit-card balance snapshots for prior HC cards.

    Revolving credit behaviour (balance carried, cash withdrawn, limit used)
    differs from instalment-loan behaviour and adds independent signal.
    """
    cc = pd.read_csv(f"{data_dir}/credit_card_balance.csv", nrows=nrows)

    # Utilisation: balance as a fraction of the credit limit. A customer running
    # consistently near their limit is stretched, and the TREND on this column
    # catches someone drawing steadily closer to it -- a classic pre-default
    # pattern that no single snapshot reveals.
    cc["CC_UTILISATION"] = (cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]
                            ).replace([np.inf, -np.inf], np.nan)

    cc, _ = _one_hot(cc)

    # Drop the per-application key: we aggregate straight to applicant level,
    # and averaging an arbitrary ID would be meaningless noise.
    cc.drop(columns=["SK_ID_PREV"], inplace=True)

    num_cols = [c for c in cc.columns if c != "SK_ID_CURR"]
    agg = cc.groupby("SK_ID_CURR").agg({c: NUM_AGGS for c in num_cols})
    agg = _flatten(agg, "CC")
    parts = [agg, cc.groupby("SK_ID_CURR").size().to_frame("CC_COUNT")]

    behaviour = ["CC_UTILISATION", "AMT_BALANCE", "AMT_DRAWINGS_ATM_CURRENT",
                 "SK_DPD"]
    behaviour = [c for c in behaviour if c in cc.columns]
    parts.append(_window_aggs(cc, "SK_ID_CURR", "MONTHS_BALANCE", behaviour,
                              "CC", unit="months").reindex(agg.index))
    parts.append(_trend(cc, "SK_ID_CURR", "MONTHS_BALANCE", behaviour,
                        "CC", unit="months").reindex(agg.index))

    agg = pd.concat(parts, axis=1)
    del cc
    gc.collect()
    return agg


def build(data_dir, nrows=None, verbose=True):
    """Run the whole pipeline and return matrices ready for XGBoost.

    Args:
        data_dir: folder holding the 10 competition CSVs.
        nrows:    cap rows read per table, for fast smoke tests.

                  CAUTION: this truncates each table to its FIRST n rows, so
                  rare pathological values may be absent from the sample. The
                  inf-producing rows that crashed the first full run were not
                  present at nrows=20000. A clean smoke test does NOT prove the
                  full run will succeed.

        verbose:  print the growing shape after each table is joined.

    Returns:
        (train_df, test_df, feature_names)
    """
    def log(msg):
        if verbose:
            print(msg, flush=True)

    # Start from the one-row-per-applicant table, then widen it table by table.
    df, n_train = application(data_dir, nrows)
    log(f"application: {df.shape}")

    # Every auxiliary table is aggregated to applicant level and left-joined.
    # LEFT join specifically: an applicant missing from a table must be kept
    # with NaN features, not dropped -- absence of history is itself a signal.
    for name, fn in [("bureau", bureau), ("previous", previous_application),
                     ("pos_cash", pos_cash), ("installments", installments),
                     ("credit_card", credit_card)]:
        agg = fn(data_dir, nrows)
        df = df.join(agg, how="left", on="SK_ID_CURR")
        log(f"+ {name}: {agg.shape[1]} features -> {df.shape}")
        del agg
        gc.collect()

    # ---- Cross-table ratios ------------------------------------------------
    # These can only be built here, after every table has been joined, because
    # each combines a column from the application table with an aggregate from
    # another source. They are a different KIND of signal from the within-table
    # ratios in application(): total debt held at OTHER lenders is meaningless
    # in isolation, but debt-to-income across all lenders describes whether this
    # applicant is already over-extended before the new loan is granted.
    def col(name):
        """Return a joined column, or NaN if that table produced no such key."""
        return df[name] if name in df.columns else np.nan

    cross = {
        # Total external debt and credit relative to this applicant's income.
        "X_BURO_DEBT_INCOME": col("BURO_AMT_CREDIT_SUM_DEBT_SUM")
        / df["AMT_INCOME_TOTAL"],
        "X_BURO_CREDIT_INCOME": col("BURO_AMT_CREDIT_SUM_SUM")
        / df["AMT_INCOME_TOTAL"],

        # External debt relative to the loan being applied for: is the new loan
        # small or large next to what they already owe elsewhere?
        "X_BURO_DEBT_CREDIT": col("BURO_AMT_CREDIT_SUM_DEBT_SUM")
        / df["AMT_CREDIT"],

        # Active external debt only -- closed loans are not a current burden.
        "X_BURO_ACTIVE_DEBT_INCOME": col("BURO_ACT_AMT_CREDIT_SUM_DEBT_SUM")
        / df["AMT_INCOME_TOTAL"],

        # Historic instalment burden vs income: what they have actually been
        # paying out each period relative to what they earn.
        "X_INS_PAYMENT_INCOME": col("INS_AMT_PAYMENT_MEAN")
        / df["AMT_INCOME_TOTAL"],

        # New annuity measured against the largest instalment they have
        # previously sustained. Well above 1 means this loan asks more of them
        # than anything in their track record.
        "X_ANNUITY_VS_MAX_INS": df["AMT_ANNUITY"]
        / col("INS_AMT_INSTALMENT_MAX"),

        # Credit-card balances carried elsewhere, against income.
        "X_CC_BALANCE_INCOME": col("CC_AMT_BALANCE_MEAN")
        / df["AMT_INCOME_TOTAL"],

        # New loan vs the typical size of loans previously granted to them.
        "X_CREDIT_VS_PREV_CREDIT": df["AMT_CREDIT"]
        / col("PREV_AMT_CREDIT_MEAN"),
    }
    df = pd.concat([df, pd.DataFrame(cross, index=df.index)], axis=1)

    # ---- Global infinity guard --------------------------------------------
    # Individual ratio features are guarded at their definition sites, but any
    # ratio added later could divide by zero on some subset of the full data
    # while looking clean on a sample. XGBoost refuses inf, so normalise once
    # here and log the count. This is a safety net, not a substitute for
    # thinking about each divide.
    num = df.select_dtypes(include=[np.number]).columns
    n_inf = int(np.isinf(df[num].to_numpy(dtype="float64", na_value=np.nan)).sum())
    if n_inf:
        log(f"replacing {n_inf} inf values with NaN")
        df[num] = df[num].replace([np.inf, -np.inf], np.nan)

    # Split back into train and test. Test rows are exactly those whose TARGET
    # is NaN, since the test CSV never had that column.
    train = df[df["TARGET"].notna()].copy()
    test = df[df["TARGET"].isna()].copy()
    del df
    gc.collect()

    # SK_ID_CURR is an identifier, not a feature -- training on it would let
    # the model memorise individuals. TARGET is the label. Both are excluded.
    drop = {"TARGET", "SK_ID_CURR", "index"}
    feats = [c for c in train.columns if c not in drop]

    # XGBoost refuses to serialise models whose feature names contain [, ] or <
    # (they collide with its internal split representation). One-hot encoding
    # readily produces such names from raw category values.
    clean = {c: c.replace("[", "_").replace("]", "_").replace("<", "_lt_")
             for c in feats}
    train = train.rename(columns=clean)
    test = test.rename(columns=clean)
    feats = [clean[c] for c in feats]

    log(f"final: train={train.shape} test={test.shape} features={len(feats)}")
    return train, test, feats
