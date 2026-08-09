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

**CV AUC 0.79501**, **private LB 0.79606** (5-fold, 307,507 rows x 1,775
features, ~24 min). Per-fold: 0.7919 / 0.8018 / 0.7918 / 0.7973 / 0.7922 —
std 0.0040, so improvements smaller than ~0.004 need LB or multi-seed
confirmation before you believe them.

1st place on the private LB was 0.8057.

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

# full run (~24 min, plus ~2 min for SHAP)
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
iterate against CV and submit only to confirm. **Trust CV over the public LB** —
the public split is only ~9,700 rows and its noise is comparable to the fold
spread, so chasing it means fitting noise.

### What the current model looks like

All plots below are from the 0.79501 run (1,775 features, 5-fold).

**Fold stability** — mean 0.7950, std 0.0040. Fold 2 sits ~0.006 above the
others, so it is an easier split rather than the model being unstable. Practical
consequence: treat gains under ~0.004 as unproven until confirmed on the LB or
across seeds.

![Fold stability](output/plots/09_fold_stability.png)

**Learning curves** — validation plateaus around round 1000 while training keeps
climbing to a final gap of ~0.10. Everything after the plateau is memorisation,
not learning. Early stopping still runs to ~2,800 rounds because validation AUC
technically drifts up by tiny amounts, so most of those trees buy almost
nothing. A larger `min_child_weight` or lower `max_depth` would close the gap;
whether that helps AUC is worth testing.

![Learning curves](output/plots/01_learning_curves.png)

**Calibration** — the model sits essentially on the diagonal across the whole
range. Predicted probabilities are usable as actual probabilities, not just as a
ranking, which matters if the scores ever feed expected-loss pricing. Note AUC
would be unchanged by any monotone transform, so this is information the metric
cannot tell you.

![Calibration](output/plots/07_calibration.png)

**Class separation** — what AUC measures, drawn directly. The blue spike below
0.05 is a large block of applicants the model confidently and correctly clears.
The orange tail reaches 0.8, so genuine high-risk cases are found. The overlap
between 0.05 and 0.25 is where the remaining error lives, and much of it is
irreducible: default is partly a random human event, and 1st place only reached
0.8057.

![Class separation](output/plots/08_score_distribution.png)

**Threshold behaviour** — ROC AUC 0.7950 is the competition metric, but the PR
curve is the operationally honest one: average precision is 0.2923 against a
0.081 base rate. Roughly 3.6x better than random, yet precision falls below 0.4
by 25% recall. Catching most defaulters means accepting many false positives —
a fact the ROC curve's large false-positive denominator hides.

![ROC and PR curves](output/plots/06_roc_pr.png)

**Feature importance and direction** — `total_gain` is the most honest single
ranking; SHAP adds the direction that no importance measure conveys (high
`EXT_SOURCE_MEAN` sits left of zero, pushing risk down).

![Importance comparison](output/plots/02_importance_comparison.png)

![SHAP beeswarm](output/plots/03_shap_beeswarm.png)

**SHAP dependence** — the shape of each top feature's effect, which the beeswarm
compresses away. `EXT_SOURCE_MEAN` is cleanly monotonic and near-linear, so the
model treats it as a smooth risk scale. `EXT_SOURCE_3` instead shows a step near
0.3 and a plateau past 0.5 — a threshold the trees discovered, beyond which more
score buys nothing. `CREDIT_TERM` is jagged and non-monotonic, the signature of
a feature the model splits on repeatedly in interaction with others rather than
reading off directly.

Vertical stripes at the far left of several panels are the NaN rows, which SHAP
places separately — visible confirmation that missingness carries its own
attribution rather than being silently imputed.

![SHAP dependence](output/plots/05_shap_dependence.png)

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
