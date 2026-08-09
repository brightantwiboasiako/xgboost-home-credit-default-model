# Home Credit Default Risk — XGBoost

Kaggle competition ([link](https://www.kaggle.com/competitions/home-credit-default-risk)),

## Layout

```
data/      10 CSVs, 2.7 GB (gitignored)
src/
  features.py     table aggregation -> feature matrix
  train.py        stratified 5-fold CV + submission
  diagnostics.py  evaluation and validation plots
  tune.py         Optuna hyperparameter search + analysis plots
output/
  submission_*.csv     one per run, CV score in the filename
  oof.npy              out-of-fold predictions (input for blending)
  importance_*.csv     weight / gain / total_gain, and mean |SHAP|
  best_params.csv      winner of the last tuning run
  plots/               all figures
```

## Current result

**CV AUC 0.7932** (5-fold, 307,507 rows x 1,674 features, ~23 min).
Per-fold: 0.7887 / 0.7988 / 0.7913 / 0.7955 / 0.7916 — spread of 0.010, so
improvements smaller than ~0.003 are within fold noise and need multiple seeds
to confirm.

## Data model

| Table | Grain | Key | Rows |
|---|---|---|---|
| `application_{train,test}` | one per applicant | `SK_ID_CURR` | 307,511 / 48,744 |
| `bureau` | one per external loan | `SK_ID_CURR`, `SK_ID_BUREAU` | 1.7M |
| `bureau_balance` | monthly per external loan | `SK_ID_BUREAU` | 27M |
| `previous_application` | one per prior HC application | `SK_ID_PREV`, `SK_ID_CURR` | 1.7M |
| `POS_CASH_balance` | monthly per prior loan | `SK_ID_PREV` | 10M |
| `installments_payments` | one per installment paid | `SK_ID_PREV` | 13.6M |
| `credit_card_balance` | monthly per prior card | `SK_ID_PREV` | 3.8M |

`bureau_balance` joins through `bureau` to reach an applicant — it is the only
table without a direct `SK_ID_CURR`.

## Setup

Dependencies are declared in `pyproject.toml` and installed into a uv-managed
`.venv` (native arm64, Python 3.12):

```bash
uv sync
```

Then either activate the venv or call it directly:

```bash
source .venv/bin/activate   # then plain `python ...`
# or, without activating:
uv run python src/train.py
```

All commands below assume one of those.

## Running

```bash
# fast smoke test (~30s) — exercises the whole pipeline including plots
python src/train.py --nrows 30000 --folds 3 --rounds 150 --shap-sample 800

# full run (~23 min)
python src/train.py

# skip diagnostics while iterating on features
python src/train.py --no-plots

# hyperparameter search, then retrain with the winner
python src/tune.py --trials 40 --nrows 60000
python src/train.py --use-tuned
```

Target ~8.07% positive rate, so stratified folds are used and accuracy is
meaningless as a metric.

> `--nrows` truncates each table to its first N rows, so rare pathological
> values may be absent. A clean smoke test does **not** guarantee the full run
> succeeds — the `inf` crash that broke the first full run was invisible at
> `--nrows 20000`.

## Submitting to Kaggle

The token at `~/.kaggle/access_token` is the newer Bearer format (`KGAT…`), not
the old `kaggle.json` username/key pair. The CLI reads it from an env var:

```bash
export KAGGLE_API_TOKEN=$(tr -d '\n' < ~/.kaggle/access_token)

# submit (train.py names each file with its CV score)
kaggle competitions submit \
  -c home-credit-default-risk \
  -f output/submission_0.79316.csv \
  -m "XGBoost, 1775 features incl. recency/trend/cross-table"

# scores appear after ~20s
kaggle competitions submissions -c home-credit-default-risk
```

Use `.venv/bin/kaggle` if the venv is not activated. Downloading the data needs
the same env var, plus a one-time rules acceptance on the competition page —
without it the API returns HTTP 403.

### Reading the result

The competition closed in 2018, so late submissions **score but stay
unranked**. Both figures are still computed against the real hidden labels:
public LB uses ~20% of the test set, private LB the other ~80%.

| Run | CV | Public LB | Private LB |
|---|---|---|---|
| Baseline (1,674 features) | 0.79316 | 0.79506 | 0.79345 |
| + time-aware features (1,775) | 0.79501 | 0.79754 | **0.79606** |

The time-aware features gained **+0.0026 on the private LB** — confirmed on
held-out data, not just CV. `INS_DPD_TREND` (slope of payment lateness over
time) ranks 9th of 1,775 features, so deterioration in payment behaviour was
genuinely missing from the lifetime aggregates.

CV tracks the private LB to within 0.001 and slightly understates it, so
iterate against CV and submit only to confirm.

CV and private LB agree to 0.0003, so **trust CV over the public LB**. The
public split is only ~9,700 rows and its noise is comparable to the 0.010 fold
spread — chasing it means fitting noise.

## Diagnostics

`train.py` writes these to `output/plots/` unless `--no-plots` is passed.

| Plot | Question it answers |
|---|---|
| `01_learning_curves` | Overfitting? Stopped too early? Train/valid gap per fold |
| `02_importance_comparison` | weight vs gain vs total_gain, side by side |
| `03_shap_beeswarm` | Which features matter, and in which **direction** |
| `04_shap_bar` | Mean \|SHAP\| ranking, comparable to gain |
| `05_shap_dependence` | Shape of each top feature's effect |
| `06_roc_pr` | ROC (the metric) and PR (honest under imbalance) |
| `07_calibration` | Are probabilities meaningful, or only rankings? |
| `08_score_distribution` | Class separation — what AUC measures, drawn |
| `09_fold_stability` | Is the CV number trustworthy? |

`tune.py` adds:

| Plot | Question it answers |
|---|---|
| `10_tuning_history` | Has the search converged? |
| `11_param_importance` | Which parameters actually matter |
| `12_slice` | Score vs value per parameter; catches too-narrow ranges |
| `13_parallel_coordinate` | Which **combinations** work |
| `14_contour` | Pairwise interaction surfaces |

Optuna's four are interactive HTML; open them in a browser.

**Reading the importance plots together** is the point. `weight` is biased
toward high-cardinality continuous features (more candidate split points).
`gain` says how much a feature helps when used. `total_gain` combines both and
is usually the most honest single ranking. SHAP then adds the direction that
none of the three convey.

## Time-aware features

The plain aggregates flatten a customer's whole history into one number, which
hides the most predictive thing in the data: someone who paid perfectly for four
years and started slipping last quarter has the same lifetime mean as a steady
payer. Three feature groups address that.

**Recency windows** (`*_MEDIAN_3M`, `_6M`, `_12M`, `_24M`) recompute behavioural
columns over trailing windows. **Median, not mean** — payment tables are
heavy-tailed, and one settlement payment or a single 300-day delinquency drags a
mean away from typical behaviour, especially in a 3-month window holding only a
few rows.

**Window deltas** (`*_DELTA_3_6M` etc.) subtract an older window from a newer
one. Positive on a lateness column means the customer is slipping.

**Trends** (`*_TREND`) fit a least-squares slope against time per customer, so
direction of travel becomes one number. Computed closed-form as
covariance/variance rather than `polyfit` per group — the latter is
prohibitively slow across ~300k groups. Customers with fewer than 3
observations get NaN, since a slope through two points is noise.

**Cross-table ratios** (`X_*`) combine an application column with an aggregate
from another table. Total external debt is meaningless alone; debt-to-income
across all lenders says whether the applicant is already over-extended. These
are built in `build()` after all joins, since they span tables.

| Group | Count | Coverage (full data) |
|---|---|---|
| Window medians | 44 | varies by table |
| Window deltas | 33 | varies by table |
| Trends | 11 | varies by table |
| Cross-table ratios | 8 | 28–95% |

Feature count: 1,674 → **1,775**. Build time ~65s.

## Modelling notes

- **`DAYS_EMPLOYED == 365243`** is a missing-value sentinel. Left in place it
  becomes a ~1000-year outlier that distorts every split. Same sentinel appears
  in five `previous_application` day columns.
- **NaNs are not filled.** XGBoost learns a default direction per split, and
  "no bureau history" is itself predictive.
- **`EXT_SOURCE_1/2/3`** are external credit scores and the strongest features
  in the dataset; their mean/std/product add signal over the raw columns.
- **Ratios beat raw amounts** — the same loan carries different risk at
  different incomes, so credit/income, annuity/income and credit term are built
  explicitly.

## Benchmarks

| Score | What |
|---|---|
| ~0.74 | application table only |
| ~0.78 | + aggregated auxiliary tables |
| 0.80+ | competitive |
| 0.805 | 1st place private LB |

The gap from 0.78 to 0.80 is where the real work is: payment-history time
windows, per-applicant trends, and stacking.

## Next steps

- Recency-weighted aggregates (last 6/12 months of installments separately)
- Trend features — is the applicant's payment behaviour deteriorating?
- LightGBM + CatBoost, blended on the saved OOF predictions
- Bayesian hyperparameter search over `max_depth`, `min_child_weight`, `eta`
