"""Evaluation and validation plots for the Home Credit XGBoost model.

WHAT EACH PLOT IS FOR
---------------------
Every function here answers one specific question you cannot answer from a
single CV number:

  learning_curves        Is the model overfitting, and when did it stop learning?
  importance_comparison  Which features matter -- and do the three importance
                         types agree? (they often do not, and that is the point)
  shap_summary           Which features matter, and in WHICH DIRECTION?
  roc_pr_curves          How does the model behave across all thresholds?
  calibration            Are the predicted probabilities trustworthy as
                         probabilities, or only as a ranking?
  score_distribution     Does the model actually separate the two classes?
  fold_stability         Is the CV score reliable, or driven by one lucky fold?

All figures are written to output/plots/ as PNG. Nothing here calls plt.show(),
so the module is safe to run headless.
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: render to file, never a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (auc, precision_recall_curve, roc_auc_score,
                             roc_curve)

# Consistent styling so plots read as one set rather than seven unrelated charts.
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colour-blind-safe palette (Okabe-Ito). Blue/orange carry train/valid meaning
# consistently across every figure in this module.
C_TRAIN = "#0072B2"
C_VALID = "#D55E00"
C_ACCENT = "#009E73"
C_MUTED = "#999999"


def _save(fig, out_dir, name):
    """Write a figure and close it, so long runs do not leak memory."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path}", flush=True)
    return path


# ---------------------------------------------------------------------------
# 1. Learning curves
# ---------------------------------------------------------------------------
def learning_curves(histories, out_dir, best_iters=None):
    """Plot train vs validation AUC per boosting round, for every fold.

    HOW TO READ IT
    The two curves separating is normal and expected -- training AUC always
    climbs higher than validation. What matters is the GAP and the SHAPE:

      - validation still rising at the end  -> stopped too early, raise --rounds
      - validation flat while train climbs  -> overfitting; the extra trees are
                                               memorising, not learning
      - large gap from the very start       -> model too complex for the signal
                                               (lower max_depth, raise
                                               min_child_weight)

    Args:
        histories: list (one per fold) of the dict returned by xgb.train via
                   evals_result, i.e. {"train": {"auc": [...]}, "valid": {...}}.
        best_iters: optional list of early-stopping iterations, marked with a
                   vertical line so you can see where training actually halted.
    """
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4), squeeze=False)

    for i, (hist, ax) in enumerate(zip(histories, axes[0])):
        tr = hist["train"]["auc"]
        va = hist["valid"]["auc"]
        rounds = np.arange(len(tr))

        ax.plot(rounds, tr, color=C_TRAIN, lw=1.4, label="train")
        ax.plot(rounds, va, color=C_VALID, lw=1.4, label="valid")

        # Shading the gap makes overfitting visible at a glance.
        ax.fill_between(rounds, va, tr, color=C_MUTED, alpha=0.18)

        if best_iters is not None:
            ax.axvline(best_iters[i], color=C_ACCENT, ls="--", lw=1.1,
                       label=f"best={best_iters[i]}")

        gap = tr[-1] - va[-1]
        ax.set_title(f"fold {i + 1}  (final gap {gap:.3f})")
        ax.set_xlabel("boosting round")
        if i == 0:
            ax.set_ylabel("AUC")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Learning curves — train vs validation AUC per round", y=1.02)
    return _save(fig, out_dir, "01_learning_curves.png")


# ---------------------------------------------------------------------------
# 2. Feature importance, three ways
# ---------------------------------------------------------------------------
def importance_comparison(model, out_dir, top_n=25):
    """Compare XGBoost's three importance measures side by side.

    WHY THREE, AND WHY THEY DISAGREE
      weight     - how many times the feature was split on. Biased toward
                   high-cardinality continuous features, which offer many
                   possible split points regardless of usefulness.
      gain       - average loss reduction per split. Answers "when this feature
                   IS used, how much does it help?"
      total_gain - gain summed over all splits (gain x weight). Usually the most
                   honest single ranking: it rewards features that help a lot
                   AND are used often.

    A feature ranking high on gain but low on total_gain is a specialist: very
    useful in the rare cases it applies. High weight but low gain is the
    opposite -- split on constantly, contributing little each time.
    """
    types = ["weight", "gain", "total_gain"]
    scores = {t: pd.Series(model.get_score(importance_type=t)) for t in types}

    # Rank by total_gain and show the same feature set across all three panels,
    # so rows line up and disagreements are visible horizontally.
    order = scores["total_gain"].sort_values(ascending=False).head(top_n).index[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, max(5, top_n * 0.28)), sharey=True)
    for ax, t in zip(axes, types):
        vals = scores[t].reindex(order).fillna(0)
        ax.barh(range(len(order)), vals.values, color=C_TRAIN, alpha=0.85)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f[:38] for f in order], fontsize=7)
        ax.set_title(t)
        ax.set_xlabel(t)

    fig.suptitle(f"Feature importance — top {top_n} by total_gain", y=1.01)
    path = _save(fig, out_dir, "02_importance_comparison.png")

    # Save the full table too: the plot shows 25 features, the CSV has all of
    # them, which is what you actually need for feature selection.
    full = pd.DataFrame(scores).sort_values("total_gain", ascending=False)
    full.to_csv(os.path.join(out_dir, "..", "importance_all_types.csv"))
    return path


# ---------------------------------------------------------------------------
# 3. SHAP
# ---------------------------------------------------------------------------
def shap_summary(model, X, out_dir, sample=4000, top_n=25, seed=42):
    """Global SHAP plots: which features matter, and in which direction.

    WHY SHAP ADDS SOMETHING IMPORTANCE DOES NOT
    Gain tells you a feature was useful but not WHICH WAY it pushed the
    prediction. SHAP assigns every feature a signed contribution for every
    individual row, so you learn e.g. "high EXT_SOURCE_MEAN pushes default risk
    DOWN" -- direction and magnitude, per applicant, aggregated into a global
    picture.

    Two plots are produced:
      beeswarm - one dot per applicant per feature. Colour is the feature's
                 value (red high / blue low), x-position is its SHAP value.
                 Red dots on the right means "high values increase risk".
      bar      - mean |SHAP| per feature: a clean magnitude-only ranking that
                 is directly comparable with the gain chart.

    Args:
        sample: rows to explain. TreeSHAP is exact but costs roughly
                O(trees x depth^2) per row, so a few thousand rows give a
                stable global picture in a fraction of the full-data time.
    """
    import shap

    # Random rather than head(): the frame may carry ordering from the source
    # CSVs, and a biased sample would give a biased explanation.
    Xs = X.sample(n=min(sample, len(X)), random_state=seed)

    # TreeExplainer exploits tree structure for exact SHAP values -- no
    # sampling approximation, unlike KernelExplainer.
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)

    fig = plt.figure(figsize=(9, max(5, top_n * 0.3)))
    shap.summary_plot(sv, Xs, max_display=top_n, show=False, plot_size=None)
    plt.title("SHAP beeswarm — value and direction of each feature's effect")
    beeswarm = _save(fig, out_dir, "03_shap_beeswarm.png")

    fig = plt.figure(figsize=(9, max(5, top_n * 0.3)))
    shap.summary_plot(sv, Xs, plot_type="bar", max_display=top_n, show=False,
                      plot_size=None)
    plt.title("SHAP importance — mean |SHAP| per feature")
    _save(fig, out_dir, "04_shap_bar.png")

    # Persist mean |SHAP| so it can be compared numerically against gain.
    pd.Series(np.abs(sv).mean(axis=0), index=Xs.columns) \
        .sort_values(ascending=False) \
        .to_csv(os.path.join(out_dir, "..", "shap_importance.csv"),
                header=["mean_abs_shap"])
    return beeswarm


def shap_dependence(model, X, features, out_dir, sample=4000, seed=42):
    """Per-feature SHAP dependence plots: the shape of each effect.

    A dependence plot puts the feature's value on x and its SHAP value on y,
    revealing whether the relationship is linear, monotonic, or has a threshold
    the model discovered. This is where you catch non-obvious behaviour, e.g.
    risk that only rises past a specific debt-to-income point.
    """
    import shap

    Xs = X.sample(n=min(sample, len(X)), random_state=seed)
    sv = shap.TreeExplainer(model).shap_values(Xs)

    feats = [f for f in features if f in Xs.columns][:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feat in zip(axes.ravel(), feats):
        shap.dependence_plot(feat, sv, Xs, ax=ax, show=False,
                             interaction_index=None, dot_size=5, alpha=0.4)
        ax.set_title(feat[:40], fontsize=9)
    for ax in axes.ravel()[len(feats):]:
        ax.set_visible(False)

    fig.suptitle("SHAP dependence — shape of each feature's effect", y=1.0)
    return _save(fig, out_dir, "05_shap_dependence.png")


# ---------------------------------------------------------------------------
# 4. Threshold behaviour and probability quality
# ---------------------------------------------------------------------------
def roc_pr_curves(y, oof, out_dir):
    """ROC and precision-recall curves from out-of-fold predictions.

    WHY BOTH
    ROC is the competition metric, but it is optimistic under class imbalance:
    the false-positive rate has a large denominator (the 92% non-defaulters), so
    even many false positives barely move it. The PR curve uses precision, whose
    denominator is only the flagged cases, so it shows the operational reality
    of a 8% base rate far more honestly.

    The PR baseline is the positive rate itself (~0.081), not 0.5.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    fpr, tpr, _ = roc_curve(y, oof)
    ax1.plot(fpr, tpr, color=C_TRAIN, lw=1.6,
             label=f"AUC = {roc_auc_score(y, oof):.4f}")
    ax1.plot([0, 1], [0, 1], color=C_MUTED, ls="--", lw=1, label="random")
    ax1.set_xlabel("false positive rate")
    ax1.set_ylabel("true positive rate")
    ax1.set_title("ROC — the competition metric")
    ax1.legend(loc="lower right", fontsize=8)

    prec, rec, _ = precision_recall_curve(y, oof)
    base = y.mean()
    ax2.plot(rec, prec, color=C_VALID, lw=1.6,
             label=f"AP = {auc(rec, prec):.4f}")
    ax2.axhline(base, color=C_MUTED, ls="--", lw=1,
                label=f"base rate = {base:.3f}")
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.set_title("Precision-Recall — honest under imbalance")
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("Threshold behaviour (out-of-fold predictions)", y=1.02)
    return _save(fig, out_dir, "06_roc_pr.png")


def calibration(y, oof, out_dir, bins=20):
    """Are predicted probabilities meaningful as probabilities?

    Bin predictions and compare each bin's mean prediction against the observed
    default rate in that bin. On the diagonal means "when the model says 30%,
    about 30% actually default" -- i.e. the score can be used for expected-loss
    pricing, not just ranking.

    AUC is invariant to any monotone transform, so a model can score well and
    still be badly calibrated. This plot is how you find that out.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    true_p, pred_p = calibration_curve(y, oof, n_bins=bins, strategy="quantile")
    ax1.plot(pred_p, true_p, "o-", color=C_TRAIN, lw=1.5, ms=4, label="model")
    ax1.plot([0, max(pred_p)], [0, max(pred_p)], color=C_MUTED, ls="--", lw=1,
             label="perfectly calibrated")
    ax1.set_xlabel("mean predicted probability")
    ax1.set_ylabel("observed default rate")
    ax1.set_title("Calibration (quantile bins)")
    ax1.legend(fontsize=8)

    ax2.hist(oof, bins=60, color=C_TRAIN, alpha=0.8)
    ax2.axvline(y.mean(), color=C_VALID, ls="--", lw=1.2,
                label=f"base rate = {y.mean():.3f}")
    ax2.set_xlabel("predicted probability")
    ax2.set_ylabel("count")
    ax2.set_title("Prediction distribution")
    ax2.legend(fontsize=8)
    ax2.set_yscale("log")  # log scale or the low-risk spike hides everything

    fig.suptitle("Probability quality", y=1.02)
    return _save(fig, out_dir, "07_calibration.png")


def score_distribution(y, oof, out_dir):
    """Predicted-score distributions for defaulters vs non-defaulters.

    This is what AUC measures, drawn directly: AUC is the probability that a
    random defaulter scores above a random non-defaulter, so it is exactly the
    degree to which these two distributions fail to overlap. Densities are
    normalised separately because the classes differ ~11x in size.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for label, colour, name in [(0, C_TRAIN, "repaid"),
                                (1, C_VALID, "defaulted")]:
        vals = oof[y == label]
        ax.hist(vals, bins=60, density=True, alpha=0.55, color=colour,
                label=f"{name} (n={len(vals):,})")
    ax.set_xlabel("predicted probability of default")
    ax.set_ylabel("density")
    ax.set_title(f"Class separation — AUC {roc_auc_score(y, oof):.4f}")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "08_score_distribution.png")


def fold_stability(fold_scores, out_dir):
    """Per-fold AUC spread — is the CV number trustworthy?

    A tight spread means the estimate is reliable and small improvements are
    real. A wide spread (std approaching the improvements you are chasing) means
    you cannot distinguish a genuine gain from fold noise, and should compare
    models across several seeds before believing any difference.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4))
    idx = np.arange(1, len(fold_scores) + 1)
    mean, std = np.mean(fold_scores), np.std(fold_scores)

    ax.bar(idx, fold_scores, color=C_TRAIN, alpha=0.85, width=0.6)
    ax.axhline(mean, color=C_VALID, ls="--", lw=1.3,
               label=f"mean = {mean:.4f}")
    ax.fill_between([0.5, len(fold_scores) + 0.5], mean - std, mean + std,
                    color=C_VALID, alpha=0.12, label=f"±1 std = {std:.4f}")

    for i, s in zip(idx, fold_scores):
        ax.text(i, s, f"{s:.4f}", ha="center", va="bottom", fontsize=8)

    # Zoom the y-range; starting at 0 would make every fold look identical.
    ax.set_ylim(min(fold_scores) - 4 * std, max(fold_scores) + 4 * std)
    ax.set_xlim(0.5, len(fold_scores) + 0.5)
    ax.set_xticks(idx)
    ax.set_xlabel("fold")
    ax.set_ylabel("validation AUC")
    ax.set_title("Fold stability")
    ax.legend(fontsize=8)
    return _save(fig, out_dir, "09_fold_stability.png")


def run_all(model, X, y, oof, fold_scores, histories, best_iters, out_dir,
            shap_sample=4000):
    """Generate every diagnostic plot. Called at the end of a training run."""
    print("\ngenerating diagnostics...", flush=True)
    learning_curves(histories, out_dir, best_iters)
    importance_comparison(model, out_dir)
    roc_pr_curves(y, oof, out_dir)
    calibration(y, oof, out_dir)
    score_distribution(y, oof, out_dir)
    fold_stability(fold_scores, out_dir)

    # SHAP last: it is by far the slowest step, so a failure here still leaves
    # every cheaper plot already written to disk.
    try:
        shap_summary(model, X, out_dir, sample=shap_sample)
        top = pd.Series(model.get_score(importance_type="total_gain")) \
            .sort_values(ascending=False).head(6).index.tolist()
        shap_dependence(model, X, top, out_dir, sample=shap_sample)
    except Exception as e:
        print(f"  SHAP skipped: {type(e).__name__}: {e}", flush=True)
