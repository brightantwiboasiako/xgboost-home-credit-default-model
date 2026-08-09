# Home Credit Default Risk - an XGBoost walkthrough

Kaggle Competition [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)

**Result: CV AUC 0.79501, private Kaggle Leaderboard (LB) 0.79606.** First place was 0.8057.

---

## Contents

- [1. The problem](#1-the-problem)
- [2. The data: seven tables, one row per applicant](#2-the-data-seven-tables-one-row-per-applicant)
  - [Two data traps worth naming](#two-data-traps-worth-naming)
- [3. Feature engineering](#3-feature-engineering)
  - [Hand-built ratios](#hand-built-ratios)
  - [Time-aware features - the part that moved the score](#time-aware-features---the-part-that-moved-the-score)
- [4. The model](#4-the-model)
  - [Hyperparameter search](#hyperparameter-search)
- [5. Results](#5-results)
  - [How much of a difference is real?](#how-much-of-a-difference-is-real)
  - [Is it overfitting?](#is-it-overfitting)
  - [What the score means operationally](#what-the-score-means-operationally)
  - [What the model learned](#what-the-model-learned)
- [6. The code](#6-the-code)
  - [Setup](#setup)
  - [Running](#running)
  - [The diagnostics](#the-diagnostics)
- [7. Reading the plots](#7-reading-the-plots)
  - [`01_learning_curves`](#01_learning_curves---train-vs-validation-auc-per-round)
  - [`02_importance_comparison`](#02_importance_comparison---weight-vs-gain-vs-total_gain)
  - [`03_shap_beeswarm`](#03_shap_beeswarm---direction-not-just-magnitude)
  - [`04_shap_bar`](#04_shap_bar---mean-shap-ranking)
  - [`05_shap_dependence`](#05_shap_dependence---the-shape-of-each-effect)
  - [`06_roc_pr`](#06_roc_pr---the-metric-and-the-honest-one)
  - [`07_calibration`](#07_calibration---are-the-probabilities-real)
  - [`08_score_distribution`](#08_score_distribution---what-auc-measures-drawn)
  - [`09_fold_stability`](#09_fold_stability---is-the-cv-number-trustworthy)
  - [`10_tuning_history`](#10_tuning_history---has-the-search-converged)

---

## 1. The problem

Home Credit lends to people with little or no traditional credit history. The
task is to predict, for each loan applicant, whether they will default - scored
by **ROC AUC** on a hidden test set.

Two facts about the target drive nearly every decision downstream:

**Only 8.07% of applicants default.** Accuracy is therefore worthless as a
metric: predicting "nobody defaults" scores 91.9% and is useless. AUC instead
measures *ranking* quality - given a random defaulter and a random non-defaulter,
how often does the model score the defaulter higher? Because only the ranking
matters, predictions are never thresholded into 0/1; the submission carries raw
probabilities and only their order affects the score.

**The signal is genuinely weak.** Default is partly a random human event - job
loss, illness, divorce. The winning solution reached 0.8057, and the gap between
a good model (0.79) and the world's best (0.805) is about 0.015 AUC. That
ceiling is the context for every result below.

## 2. The data: seven tables, one row per applicant

The central difficulty is structural, not statistical. XGBoost needs a
rectangular matrix with **exactly one row per applicant**, but only one of the
seven tables is shaped that way.

| Table | Grain | Key | Rows |
|---|---|---|---|
| `application_{train,test}` | one per applicant | `SK_ID_CURR` | 307,511 / 48,744 |
| `bureau` | one per external loan | `SK_ID_CURR`, `SK_ID_BUREAU` | 1.7M |
| `bureau_balance` | monthly per external loan | `SK_ID_BUREAU` | 27M |
| `previous_application` | one per prior HC application | `SK_ID_PREV`, `SK_ID_CURR` | 1.7M |
| `POS_CASH_balance` | monthly per prior loan | `SK_ID_PREV` | 10M |
| `installments_payments` | one per installment paid | `SK_ID_PREV` | 13.6M |
| `credit_card_balance` | monthly per prior card | `SK_ID_PREV` | 3.8M |

2.7 GB of CSV, and the target lives only in `application_train`. Every other
table has many rows per applicant - one per external loan, one per monthly
snapshot, one per payment made.

So every auxiliary table goes through the same three steps ([features.py](src/features.py)):

1. **one-hot encode** its categorical columns, so they can be averaged
2. **`groupby(SK_ID_CURR).agg([...])`**, collapsing to one row per applicant
3. **left-join** the result onto the application frame

This *aggregate-then-join* pattern is the whole competition. `bureau_balance` is
the one exception requiring two levels - it has no `SK_ID_CURR` at all, so it
aggregates to the loan first, joins through `bureau`, then aggregates again to
the applicant.

**Why aggregation captures signal.** A single applicant may have 30 installment
payments. Any one payment says little, but `mean(days_late)` says whether they
habitually pay late and `max(days_late)` gives their worst-ever delinquency.
Those summaries are what the model can learn from. Five statistics -
mean, max, min, sum, std - are applied to most numeric columns, so each source
column expands into five features.

**Left joins, and NaNs are never filled.** An applicant absent from a table
(no credit history at all) keeps their row with NaN features. XGBoost learns a
default branch direction for missing values at every split, and *"this person has
no bureau record"* is itself a risk signal that imputation would erase.

### Two data traps worth naming

**`DAYS_EMPLOYED == 365243`** is a missing-value sentinel. The column holds
negative day offsets from the application date, so left in place that value
reads as roughly **+1000 years** - a massive outlier dragging split thresholds
away from the real data. The same sentinel appears in five `previous_application`
day columns. All are replaced with NaN.

**Division by zero produces `inf`, and XGBoost refuses to train on it.**
`AMT_CREDIT` is 0 on ~337k cancelled applications, and 290 installment rows have
`AMT_INSTALMENT == 0`. Both feed ratio features. This crashed the first full
training run - and, importantly, was *invisible* at `--nrows 20000`, because
`--nrows` truncates each table to its first N rows and the pathological rows sat
further down. Each divide is now guarded at its definition site, with a global
`inf → NaN` sweep in `build()` as a safety net.

## 3. Feature engineering

Two layers, built in [features.py](src/features.py).

### Hand-built ratios

Raw amounts are weak predictors because the same loan means different things at
different income levels - a $50k loan is routine on a $200k income and alarming
on a $20k one. Encoding ratios directly saves the model from rediscovering them
through splits: `CREDIT_INCOME_RATIO`, `ANNUITY_INCOME_RATIO`, `CREDIT_TERM`
(annuity/credit, approximating loan term), `EMPLOYED_BIRTH_RATIO` (fraction of
life spent in the current job), `INCOME_PER_PERSON`, `GOODS_CREDIT_RATIO`.

`EXT_SOURCE_1/2/3` are normalised external credit scores and consistently the
strongest features in this dataset. Combining them adds information beyond the
three raw columns: the **mean** is a more stable consensus, the **std** flags
applicants the agencies disagree about, and the **product** is a sharp
non-linear interaction.

### Time-aware features - the part that moved the score

Lifetime aggregates flatten a customer's whole history into one number, which
hides the most predictive thing in the data: **someone who paid perfectly for
four years and started slipping last quarter has the same lifetime mean as a
steady payer.** Four feature groups address that.

**Recency windows** (`*_MEDIAN_3M`, `_6M`, `_12M`, `_24M`) recompute behavioural
columns over trailing windows. Median, not mean - payment tables are
heavy-tailed, and one settlement payment or a single 300-day delinquency drags a
mean away from typical behaviour, especially in a 3-month window holding only a
few rows.

**Window deltas** (`*_DELTA_3_6M`) subtract an older window from a newer one.
Positive on a lateness column means the customer is slipping. A raw window
median says what someone did recently; recent-minus-older says whether they are
getting *better or worse*, which is the actual risk signal.

**Trends** (`*_TREND`) fit a least-squares slope against time per customer, so
direction of travel becomes one number. Computed closed-form as
covariance/variance rather than `np.polyfit` per group - the latter is
prohibitively slow across ~300k groups, while this is a handful of vectorised
groupby passes. Customers with fewer than 3 observations get NaN, since a slope
through two points is noise.

**Cross-table ratios** (`X_*`) combine an application column with an aggregate
from another table, so they can only be built after all joins. Total external
debt is meaningless alone; debt-to-income *across all lenders* says whether the
applicant is already over-extended before this loan is granted. Likewise
`X_ANNUITY_VS_MAX_INS` - the new annuity against the largest installment they
have ever sustained. Well above 1 means this loan asks more of them than
anything in their track record.

| Group | Count |
|---|---|
| Window medians | 44 |
| Window deltas | 33 |
| Trends | 11 |
| Cross-table ratios | 8 |

**Feature count: 1,674 → 1,775.** Build time ~65s.

## 4. The model

XGBoost, 5-fold stratified CV ([train.py](src/train.py)).

```python
"objective":        "binary:logistic",  # calibrated probabilities in [0,1]
"eval_metric":      "auc",              # early-stop on the competition metric
"tree_method":      "hist",             # binned splits; ~1,775 features x 300k rows
"eta":              0.02,               # low LR + many rounds + early stopping
"max_depth":        6,                  # ~6-way interactions; deeper memorises
"min_child_weight": 40,                 # main anti-overfit lever at an 8% positive rate
"subsample":        0.85,
"colsample_bytree": 0.7,                # decorrelates near-duplicate aggregates
"reg_alpha":        0.1,                # L1: weak weights to exactly zero
"reg_lambda":       1.0,
```

The choices that matter most here:

**Low `eta` with early stopping.** Each tree contributes a small correction, so
the ensemble approaches the signal gradually rather than chasing noise. On data
this noisy, a higher learning rate overfits before the aggregated features have
paid off. `num_boost_round=5000` is an upper bound, not a target - training ends
when validation AUC stops improving for 200 rounds, so the data picks the tree
count.

**`min_child_weight=40`.** With only ~8% positives, this is the main guard
against carving out tiny leaves that fit a handful of defaulters by coincidence.

**`colsample_bytree=0.7`** matters more than usual here, because many aggregate
features are near-duplicates (`INS_DPD_MEAN` vs `INS_DBD_MEAN`). Without column
subsampling every tree would keep reaching for the same dominant columns.

**Stratified folds** preserve the ~8% default rate inside every fold. Plain
`KFold` would let fold base rates drift apart by chance, adding variance to the
CV estimate and making folds non-comparable.

**The out-of-fold vector is the key artefact.** Each row is predicted by the one
model that did *not* train on it, so `roc_auc_score(y, oof)` is an honest
estimate over the full training set. It is saved to `output/oof.npy` because it
is also exactly what a later blending layer needs. Test predictions are averaged
across the five fold models - a mild ensemble that reliably beats any single
fold's model.

### Hyperparameter search

[tune.py](src/tune.py) runs Optuna's TPE sampler. A grid over 6 parameters at 4
values each is 4,096 fits, which is impossible on a laptop at several minutes
per fit; Bayesian search models which regions produced good scores and
concentrates trials there.

A 40-trial run on 60k rows took **24 minutes**, of which **15 trials were
pruned** - MedianPruner reports the running mean after each fold and aborts
configurations already below the median, which is where most of the wall-clock
saving comes from. Best trial (#33, of 25 completed) reached 0.757975 on the
reduced setup, against 0.752723 for the worst - a spread of only **0.005**.

That narrow spread is the finding. **Hyperparameters were worth far less than
features here**: the time-aware feature groups moved the private LB by 0.0026
while the entire searched parameter space spanned 0.005 on a reduced setup. The
tuned configuration favours a deeper tree (`max_depth=8`) with much heavier
regularisation (`min_child_weight≈84`, `eta≈0.010`) - a different balance point
than the hand-set defaults, not a dramatically better one.

Search defaults trade fidelity for speed (3 folds, a row subsample). The
*ranking* of configurations is stable under those reductions even though
absolute AUC is lower, which is all the search needs. Re-validate the winner
with a full 5-fold run via `--use-tuned`.

## 5. Results

**CV AUC 0.79501, private LB 0.79606** - 307,507 rows x 1,775 features, ~24 min.

The competition closed in 2018, so late submissions **score but stay unranked**.
Both LB figures are still computed against the real hidden labels: public uses
~20% of the test set, private the other ~80%.

| Run | Features | CV | Public LB | Private LB |
|---|---|---|---|---|
| application table only | ~240 | ~0.74 | | |
| Baseline (all tables aggregated) | 1,674 | 0.79316 | 0.79506 | 0.79345 |
| **+ time-aware features** | **1,775** | **0.79501** | 0.79754 | **0.79606** |
| 1st place private LB | | | | 0.8057 |

The time-aware features gained **+0.0026 on the private LB** - confirmed on
held-out data, not just CV. `INS_DPD_TREND` (slope of payment lateness over
time) ranks 9th of 1,775 by gain, so deterioration in payment behaviour was
genuinely missing from the lifetime aggregates.

**CV tracks the private LB to within 0.001 and slightly understates it.** So
iterate against CV and submit only to confirm. **Trust CV over the public LB** -
the public split is only ~9,700 rows and its noise is comparable to the fold
spread, so chasing it means fitting noise.

### How much of a difference is real?

Per-fold: 0.7919 / 0.8018 / 0.7918 / 0.7973 / 0.7922 - mean 0.7950, **std
0.0040**. Fold 2 sits ~0.006 above the others, so it is an easier split rather
than the model being unstable.

The practical consequence governs every experiment: **treat gains under ~0.004
as unproven** until confirmed on the LB or across multiple seeds.

![Fold stability](output/plots/09_fold_stability.png)

### Is it overfitting?

Validation plateaus around round 1000 while training keeps climbing to a final
gap of ~0.10. Everything after the plateau is memorisation, not learning. Early
stopping still runs to ~2,800 rounds because validation AUC technically drifts
up by tiny amounts, so most of those trees buy almost nothing. A larger
`min_child_weight` or lower `max_depth` would close the gap; whether that helps
AUC is worth testing.

![Learning curves](output/plots/01_learning_curves.png)

### What the score means operationally

ROC AUC 0.7950 is the competition metric, but the PR curve is the honest one.
Average precision is **0.2923 against a 0.081 base rate** - roughly 3.6x better
than random, yet precision falls below 0.4 by 25% recall. **Catching most
defaulters means accepting many false positives**, a fact the ROC curve's large
false-positive denominator hides.

![ROC and PR curves](output/plots/06_roc_pr.png)

Drawing class separation directly shows what AUC measures. The blue spike below
0.05 is a large block of applicants the model confidently and correctly clears.
The orange tail reaches 0.8, so genuine high-risk cases are found. The overlap
between 0.05 and 0.25 is where the remaining error lives - and much of it is
irreducible.

![Class separation](output/plots/08_score_distribution.png)

The model also sits essentially **on the calibration diagonal** across the whole
range. Predicted probabilities are usable as actual probabilities, not just as a
ranking, which matters if the scores ever feed expected-loss pricing. Note that
AUC is invariant to any monotone transform, so this is information the
competition metric simply cannot tell you.

![Calibration](output/plots/07_calibration.png)

### What the model learned

Three importance measures, read together. `weight` (split count) is biased
toward high-cardinality continuous features, which offer many candidate split
points regardless of usefulness. `gain` says how much a feature helps *when
used*. `total_gain` combines both and is usually the most honest single ranking.
A feature high on gain but low on total_gain is a specialist - very useful in the
rare cases it applies.

![Importance comparison](output/plots/02_importance_comparison.png)

Top features by mean gain across folds:

| # | Feature | Gain |
|---|---|---|
| 1 | `EXT_SOURCE_MEAN` | 204.6 |
| 2 | `BURO_CREDIT_ACTIVE_CLOSED_MAX` | 90.4 |
| 3 | `EXT_SOURCE_3` | 67.8 |
| 4 | `EXT_SOURCE_2` | 56.1 |
| 5 | `CC_CC_UTILISATION_MEDIAN_3M` | 52.7 |
| 6 | `EXT_SOURCE_PROD` | 52.6 |
| 7 | `NAME_EDUCATION_TYPE_Higher education` | 50.7 |
| 8 | `PREV_CODE_REJECT_REASON_SCOFR_SUM` | 38.7 |
| 9 | `INS_DPD_TREND` | 33.4 |
| 10 | `PREV_REFUSED_RATE` | 31.3 |

Three things stand out. The **external credit scores dominate** - the engineered
`EXT_SOURCE_MEAN` outranks all three raw columns, confirming the consensus is
more stable than any single agency. **Home Credit's own prior rejections**
(`PREV_CODE_REJECT_REASON_SCOFR_SUM`, `PREV_REFUSED_RATE`) rank high, because
they encode risk assessments the lender already made about this person. And
**three of the top ten are time-aware features** (`CC_CC_UTILISATION_MEDIAN_3M`,
`INS_DPD_TREND`, plus the 6M utilisation window just below), which is the direct
evidence behind the +0.0026 gain.

Gain says a feature was useful but not *which way* it pushed the prediction.
SHAP assigns every feature a signed contribution per row, so high
`EXT_SOURCE_MEAN` visibly sits left of zero - pushing risk down.

![SHAP beeswarm](output/plots/03_shap_beeswarm.png)

The dependence plots show the *shape* of each effect, which the beeswarm
compresses away. `EXT_SOURCE_MEAN` is cleanly monotonic and near-linear, so the
model treats it as a smooth risk scale. `EXT_SOURCE_3` instead shows a step near
0.3 and a plateau past 0.5 - a threshold the trees discovered, beyond which more
score buys nothing. `CREDIT_TERM` is jagged and non-monotonic, the signature of
a feature the model splits on repeatedly *in interaction* with others rather
than reading off directly.

Vertical stripes at the far left of several panels are the NaN rows, which SHAP
places separately - visible confirmation that missingness carries its own
attribution rather than being silently imputed.

![SHAP dependence](output/plots/05_shap_dependence.png)

## 6. The code

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
  tuning_trials.csv    every trial, for comparing successive searches
  plots/               all figures
```

### Setup

Dependencies are declared in `pyproject.toml` and installed into a uv-managed
`.venv` (native arm64, Python 3.12):

```bash
uv sync
source .venv/bin/activate   # then plain `python ...`
# or, without activating: uv run python src/train.py
```

### Running

```bash
# fast smoke test (~30s) - exercises the whole pipeline including plots
python src/train.py --nrows 30000 --folds 3 --rounds 150 --shap-sample 800

# full run (~24 min, plus ~2 min for SHAP)
python src/train.py

# skip diagnostics while iterating on features
python src/train.py --no-plots

# hyperparameter search, then retrain with the winner
python src/tune.py --trials 40 --nrows 60000
python src/train.py --use-tuned
```

### The diagnostics

`train.py` writes these to `output/plots/` unless `--no-plots` is passed. Each
answers one question a single CV number cannot.

| Plot | Question it answers |
|---|---|
| `01_learning_curves` | Overfitting? Stopped too early? Train/valid gap per fold |
| `02_importance_comparison` | weight vs gain vs total_gain, side by side |
| `03_shap_beeswarm` | Which features matter, and in which **direction** |
| `04_shap_bar` | Mean \|SHAP\| ranking, comparable to gain |
| `05_shap_dependence` | Shape of each top feature's effect |
| `06_roc_pr` | ROC (the metric) and PR (honest under imbalance) |
| `07_calibration` | Are probabilities meaningful, or only rankings? |
| `08_score_distribution` | Class separation - what AUC measures, drawn |
| `09_fold_stability` | Is the CV number trustworthy? |

`tune.py` adds four interactive Optuna HTML figures - open them in a browser:

| Plot | Question it answers |
|---|---|
| `10_tuning_history` | Has the search converged? |
| `11_param_importance` | Which parameters actually matter |
| `12_slice` | Score vs value per parameter; catches too-narrow ranges |
| `13_parallel_coordinate` | Which **combinations** work |
| `14_contour` | Pairwise interaction surfaces |

SHAP runs last in `run_all()` because it is by far the slowest step, so a failure
there still leaves every cheaper plot on disk. It explains **held-out validation
rows**, never training rows.

## 7. Reading the plots

What each figure actually shows in this run, and what to conclude from it.

### `01_learning_curves` - train vs validation AUC per round

![Learning curves](output/plots/01_learning_curves.png)

Five panels, one per fold. Train (blue) climbs to ~0.895-0.900 while validation
(orange) flattens near 0.79-0.80 by roughly round 1000. Final train/valid gaps
are tight across folds: 0.104, 0.095, 0.103, 0.104, 0.099.

**Interpretation.** The plateau at ~1000 rounds against best iterations of
2916 / 2897 / 2820 / 3194 / 2683 means roughly two thirds of the trees are bought
for almost nothing - validation AUC keeps creeping up by amounts too small to
matter, so early stopping's 200-round patience never triggers. The ~0.10 gap is
memorisation, but it is *stable* memorisation: the near-identical gap and
best-iteration range across all five folds says the model is behaving
consistently, not overfitting erratically on one split. Fold 2's higher
validation curve is a property of that split, not of that model.

**What would change the picture.** A validation curve that turned *downward*
after the plateau would be genuine harmful overfitting and would demand a
smaller `max_depth` or larger `min_child_weight`. It does not. So the gap is
worth an experiment, not an emergency.

### `02_importance_comparison` - weight vs gain vs total_gain

![Importance comparison](output/plots/02_importance_comparison.png)

Top 25 features by `total_gain`, with all three measures side by side. The
disagreement between panels is the point.

**Interpretation.** `CREDIT_TERM` has the **highest weight of any feature**
(~1,450 splits, more than `EXT_SOURCE_MEAN`) but a modest gain of ~20. That is
the classic high-cardinality continuous feature: it offers many candidate split
points, so it gets chosen constantly while each individual split buys little.
`EXT_SOURCE_MEAN` is the mirror image - fewer splits than `CREDIT_TERM` but a
gain of ~215, an order of magnitude above everything else, and it dominates
`total_gain` at ~255,000.

`CC_CC_UTILISATION_MEDIAN_3M` is the specialist case worth noticing: near the
**bottom on weight** (~200 splits, lowest in the panel) yet ~50 on gain, 5th
overall. It rarely applies - only applicants with recent credit-card history -
but when it does it is decisive. Reading weight alone would have discarded it.

### `03_shap_beeswarm` - direction, not just magnitude

![SHAP beeswarm](output/plots/03_shap_beeswarm.png)

Each dot is one validation applicant; colour is the feature's value (red high,
blue low), horizontal position is that feature's signed push on the prediction.

**Interpretation.** `EXT_SOURCE_MEAN` shows textbook clean separation - red dots
(high score) sit far left to -0.9, blue dots right to +1.0. High external score
pushes predicted risk **down**, monotonically, and its spread is roughly triple
any other feature's. `CREDIT_TERM` and `GOODS_CREDIT_RATIO` run the opposite
direction: blue (low) sits right, so *shorter* terms and *lower* goods-to-credit
ratios raise predicted risk.

`DAYS_EMPLOYED` and `DAYS_BIRTH` are negative day-offsets, so red means recent -
recently employed and younger applicants push risk up, which is the expected
direction and a useful sanity check that the sentinel cleanup worked.
`INS_DPD_TREND` appears here with red (worsening lateness) pushing right, direct
visual confirmation that the trend features encode deterioration as intended.

### `04_shap_bar` - mean |SHAP| ranking

![SHAP bar](output/plots/04_shap_bar.png)

The beeswarm collapsed to average magnitude, and the useful thing is that it
**disagrees with gain**.

**Interpretation.** `EXT_SOURCE_MEAN` at ~0.34 is more than three times the next
feature (~0.10), a far more extreme concentration than gain suggested. Below it,
ranks 2-6 are nearly tied at 0.09-0.10 (`EXT_SOURCE_2`, `CREDIT_TERM`,
`GOODS_CREDIT_RATIO`, `EXT_SOURCE_3`, `AMT_ANNUITY`) - differences there are
noise, not a ranking.

Note `CREDIT_TERM` rises to 3rd here despite mediocre gain, and
`BURO_CREDIT_ACTIVE_CLOSED_MAX` - 2nd by gain - **does not appear in the top 25
at all**. Gain credits a feature at the split that used it; SHAP distributes
credit per prediction across the whole ensemble. When the two disagree this
sharply, the feature is contributing through interactions rather than
standalone splits. Trust SHAP for "what drives predictions", gain for "what the
trees found useful to split on".

### `05_shap_dependence` - the shape of each effect

![SHAP dependence](output/plots/05_shap_dependence.png)

Six panels, each plotting a feature's value against its own SHAP contribution.

**Interpretation.** Three distinct shapes appear, and they call for different
treatment:

- **`EXT_SOURCE_MEAN`** - a clean, near-linear descent from +1.0 to -0.85 with
  tight scatter. The model treats it as a smooth continuous risk scale. Nothing
  to improve here; binning it would only destroy information.
- **`EXT_SOURCE_3`** - visibly *stepped*, with a drop near 0.3 and a plateau
  past 0.5 flattening around -0.10. Beyond ~0.5 additional score buys nothing.
  That is a threshold the trees discovered, not a linear relationship.
- **`CREDIT_TERM`** - jagged, non-monotonic, vertical stripes of ±0.3 at nearly
  identical x-values. The same input value maps to opposite contributions
  depending on other features, which is the signature of a feature acting almost
  entirely **through interactions**. This explains its weight/gain split above.

`GOODS_CREDIT_RATIO` has a sharp cliff at 1.0 - the point where the loan exceeds
the value of the goods - and the isolated marks at the far left of several
panels are NaN rows, which SHAP attributes separately. Missingness is carrying
its own signal, exactly as the never-impute decision intended.

### `06_roc_pr` - the metric, and the honest one

![ROC and PR curves](output/plots/06_roc_pr.png)

**Interpretation.** The ROC curve (AUC 0.7950) rises steeply and looks strong.
The PR curve tells the operational truth: **AP 0.2923 against a 0.081 base
rate** - 3.6x better than random, but precision starts near 0.6, falls below 0.4
by ~25% recall, and reaches ~0.2 at 60% recall.

Concretely: to catch 60% of defaulters you accept roughly **four false positives
for every true one**. ROC hides this because its false-positive denominator
includes all 282,682 repaid applicants, so thousands of false positives barely
move the x-axis. Optimise for AUC because the competition says so, but never
quote it as a business capability.

### `07_calibration` - are the probabilities real?

![Calibration](output/plots/07_calibration.png)

**Interpretation.** The reliability curve sits essentially **on the diagonal**
across the full 0.00-0.40 range, with only a slight bulge near 0.08-0.10 where
the model runs marginally hot. Applicants predicted at 0.20 default at
approximately 20%. These are usable probabilities, not just rankings - which
matters for expected-loss pricing, and is information AUC cannot report, since
AUC is invariant to any monotone transform.

The right panel (log-scale counts) shows why the score is hard-won: the mass is
crushed below 0.1, thinning past 0.4, with only a handful of applicants above
0.8. The model is confident about who is *safe* and rarely certain about who
will default.

### `08_score_distribution` - what AUC measures, drawn

![Class separation](output/plots/08_score_distribution.png)

Densities of predicted probability for repaid (blue, n=282,682) and defaulted
(orange, n=24,825).

**Interpretation.** The blue spike reaching density 15 below 0.05 is the model's
real product: a large block of applicants confidently and correctly cleared. The
orange distribution is flatter and its tail extends past 0.6, so genuine
high-risk cases do get found.

But the two distributions **overlap heavily between 0.05 and 0.25**, and that
overlap is where nearly all the error lives. Much of it is irreducible - default
is partly a random human event, and no feature set separates a borrower who
loses their job next year from an identical one who does not. This picture is
why 0.795 and the winning 0.8057 are closer than they look.

### `09_fold_stability` - is the CV number trustworthy?

![Fold stability](output/plots/09_fold_stability.png)

Per-fold: 0.7919 / 0.8018 / 0.7918 / 0.7973 / 0.7922, mean 0.7950, std 0.0040.

**Interpretation.** Folds 1, 3 and 5 are tightly clustered within 0.0004 of each
other. Fold 2 sits ~0.006 above them and fold 4 ~0.002 above - and because fold
2 is high on *both* train and validation in plot 01, it is an easier split
rather than a better model.

**This is the most operationally important plot.** With std 0.0040, an
experiment that improves CV by 0.002 has produced a result well inside the noise
of a single fold reassignment. Hence the standing rule: **treat gains under
~0.004 as unproven** until confirmed on the LB or across seeds. The +0.0026 from
time-aware features only counts because the private LB confirmed it
independently.

### `10_tuning_history` - has the search converged?

![Tuning history](output/plots/10_tuning_history.png)

25 completed trials (15 more pruned), best-so-far in orange.

**Interpretation.** The curve makes almost all of its progress by trial 12 and
then flattens - two marginal improvements after that, ending at **0.757975**
(trial 33). Total spread across completed trials is roughly 0.7527 to 0.7580, so
**the entire searched space is worth ~0.005**, and the last dozen trials bought
~0.0004 between them.

The flattening is the signal to stop: more trials in these ranges will not pay.
Compare against the feature work - the time-aware groups moved the private LB
0.0026 on their own. **Features were worth far more than hyperparameters here**,
and this plot is the evidence. Absolute values are ~0.037 below the full-run CV
because the search uses 3 folds on 60k rows; only the *ranking* needs to survive
that reduction, and it does.
