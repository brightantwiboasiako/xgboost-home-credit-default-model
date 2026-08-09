"""Train an XGBoost model on Home Credit and write a Kaggle submission.

WHAT THE METRIC DEMANDS
-----------------------
The competition is scored by ROC AUC on a target where only 8.07% of applicants
default. Two consequences drive nearly every choice in this file:

  1. Accuracy is useless here. Predicting "nobody defaults" scores 91.9%
     accuracy and is worthless. AUC instead measures RANKING quality: given a
     random defaulter and a random non-defaulter, how often does the model score
     the defaulter higher?

  2. Because only AUC matters, we never threshold predictions into 0/1. The
     submission contains raw probabilities, and only their ORDER affects score.

WHY CROSS-VALIDATION RATHER THAN ONE SPLIT
------------------------------------------
A single train/validation split gives a noisy score, and tuning against it
gradually overfits that one split. K-fold trains K models, each validated on a
portion it never saw, so every training row contributes to the estimate.

The out-of-fold (OOF) prediction vector is the key artefact: each row is
predicted by the one model that did NOT train on it. So `roc_auc_score(y, oof)`
is an honest estimate over the full training set, and the saved OOF vector is
also the input a later stacking/blending layer needs.

USAGE
-----
    python src/train.py                # full 5-fold run
    python src/train.py --nrows 20000  # fast smoke test (see caveat in build())
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import diagnostics
import features

# Paths are derived from this file's location so the script runs correctly
# regardless of the caller's working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
PLOTS = os.path.join(OUT, "plots")

PARAMS = {
    # Binary classification returning calibrated probabilities in [0, 1].
    "objective": "binary:logistic",

    # Optimise and early-stop directly on the competition metric.
    "eval_metric": "auc",

    # Histogram-based split finding: buckets continuous features into bins
    # instead of testing every distinct value. With ~1,600 features and 300k
    # rows this is dramatically faster at essentially no cost in accuracy.
    "tree_method": "hist",

    # Low learning rate + many rounds + early stopping. Each tree contributes
    # only a small correction, so the ensemble approaches the signal gradually
    # rather than chasing noise. This dataset is noisy (default is a partly
    # random human event), and higher eta overfits before the aggregated
    # features have paid off.
    "eta": 0.02,

    # Depth 6 allows roughly six-way feature interactions. Deeper trees can
    # memorise individual applicants given this many columns.
    "max_depth": 6,

    # A leaf must carry substantial total weight before a split is kept. With
    # only ~8% positives this is the main guard against carving out tiny leaves
    # that fit a handful of defaulters by coincidence.
    "min_child_weight": 40,

    # Row and column subsampling. Each tree sees 85% of rows and 70% of
    # features, which decorrelates the ensemble. Column subsampling matters
    # especially here because many aggregate features are near-duplicates
    # (e.g. INS_DPD_MEAN vs INS_DBD_MEAN); without it, every tree would keep
    # reaching for the same handful of dominant columns.
    "subsample": 0.85,
    "colsample_bytree": 0.7,

    # L1 drives weak feature weights to exactly zero; L2 shrinks large ones.
    # Both matter with ~1,600 heavily collinear aggregates.
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,

    "nthread": -1,  # use all cores
}


def run(nrows=None, folds=5, rounds=5000, early_stop=200, seed=42,
        plots=True, shap_sample=4000, use_tuned=False):
    """Build features, run K-fold CV, and write submission + diagnostics.

    Args:
        plots:       generate the diagnostic figures in output/plots/.
        shap_sample: rows to explain with SHAP (the slowest diagnostic).
        use_tuned:   load output/best_params.csv from a tune.py run instead of
                     the hand-set PARAMS above.

    Returns:
        The cross-validated AUC (float).
    """
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    # Tuned values override the defaults but never the structural settings
    # (objective, metric, tree_method), which tune.py does not search over.
    params = dict(PARAMS)
    if use_tuned:
        tuned_path = os.path.join(OUT, "best_params.csv")
        if os.path.exists(tuned_path):
            tuned = pd.read_csv(tuned_path).iloc[0].to_dict()
            tuned["max_depth"] = int(tuned["max_depth"])
            params.update(tuned)
            print(f"using tuned params from {tuned_path}:")
            for k, v in tuned.items():
                print(f"  {k:20s} {v}")
        else:
            print(f"no {tuned_path} found; run tune.py first. "
                  f"Falling back to defaults.")

    # Feature engineering lives entirely in features.py; see that module for
    # how the seven source tables collapse into this matrix.
    train, test, feats = features.build(DATA, nrows=nrows)
    y = train["TARGET"].astype(int).values
    X = train[feats]
    X_test = test[feats]

    # StratifiedKFold preserves the ~8% default rate inside every fold. Plain
    # KFold would let fold base rates drift apart purely by chance, adding
    # variance to the CV estimate and making folds non-comparable.
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    oof = np.zeros(len(train))    # each row filled by the model that skipped it
    preds = np.zeros(len(test))   # test predictions, averaged across folds
    importances = []

    # Captured per fold so diagnostics.py can draw learning curves and the
    # fold-stability chart. histories holds the full per-round AUC traces;
    # last_model is kept for the importance and SHAP plots, which need a live
    # booster rather than a saved score dict.
    histories, fold_scores, best_iters = [], [], []
    last_model, last_valid_X = None, None

    # Built once and reused: converting ~48k x 1,600 values to DMatrix is
    # expensive and the test features never change between folds.
    dtest = xgb.DMatrix(X_test, feature_names=feats)

    for i, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        # DMatrix is XGBoost's internal column-compressed format. NaNs are
        # preserved and handled natively as "missing" during training.
        dtr = xgb.DMatrix(X.iloc[tr_idx], y[tr_idx], feature_names=feats)
        dva = xgb.DMatrix(X.iloc[va_idx], y[va_idx], feature_names=feats)

        # num_boost_round is an upper bound, not a target. Early stopping ends
        # training once validation AUC has not improved for `early_stop` rounds,
        # so the effective tree count is chosen by the data rather than by us.
        #
        # evals_result captures the AUC of both sets at every round; that trace
        # is what the learning-curve plot is drawn from.
        hist = {}
        model = xgb.train(
            params, dtr,
            num_boost_round=rounds,
            evals=[(dtr, "train"), (dva, "valid")],
            early_stopping_rounds=early_stop,
            verbose_eval=500,
            evals_result=hist,
        )
        histories.append(hist)

        # After early stopping the final trees are the overfitting ones, so
        # predict using only trees up to the best validation iteration.
        best = model.best_iteration
        oof[va_idx] = model.predict(dva, iteration_range=(0, best + 1))

        # Average this fold's test predictions into the running total. Averaging
        # K models trained on different subsets is a mild ensemble and reliably
        # beats any single fold's model.
        preds += model.predict(dtest, iteration_range=(0, best + 1)) / folds

        score = roc_auc_score(y[va_idx], oof[va_idx])
        fold_scores.append(score)
        best_iters.append(best)
        print(f"fold {i}/{folds}  auc={score:.6f}  best_iter={best}", flush=True)

        # "gain" = total loss reduction this feature delivered across all its
        # splits. Preferred over the default "weight" (raw split count), which
        # merely favours high-cardinality columns.
        importances.append(
            pd.Series(model.get_score(importance_type="gain"), name=f"fold{i}"))

        # Keep the final fold's booster and its validation rows for the
        # importance and SHAP plots. One fold's model is representative for
        # explanation purposes, and holding all K would be needlessly heavy.
        last_model, last_valid_X = model, X.iloc[va_idx]

        # Free the fold's matrices before building the next pair; each is
        # several GB at full size.
        del dtr, dva

    # AUC over all OOF predictions at once. Slightly more reliable than
    # averaging the per-fold scores, since it uses one common ranking.
    cv = roc_auc_score(y, oof)
    print(f"\nCV AUC: {cv:.6f}   ({time.time() - t0:.0f}s)")

    # ---- Artefacts ---------------------------------------------------------
    # Submission format required by Kaggle: SK_ID_CURR plus a TARGET
    # probability. The CV score is embedded in the filename so that repeated
    # experiments remain comparable on disk.
    sub = pd.DataFrame({"SK_ID_CURR": test["SK_ID_CURR"].astype(int),
                        "TARGET": preds})
    sub_path = os.path.join(OUT, f"submission_{cv:.5f}.csv")
    sub.to_csv(sub_path, index=False)

    # Saved for later blending: a second model's OOF vector can be combined
    # with this one and the blend weight chosen by maximising AUC on OOF.
    np.save(os.path.join(OUT, "oof.npy"), oof)

    # Mean gain across folds; features stable across folds are the real ones.
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    imp.to_csv(os.path.join(OUT, "importance.csv"), header=["gain"])

    print(f"wrote {sub_path}")
    print("\ntop 15 features by gain:")
    print(imp.head(15).to_string())

    # Diagnostics run last so that the submission is already safely on disk
    # even if plotting fails (e.g. a missing optional dependency).
    if plots:
        diagnostics.run_all(
            model=last_model,
            X=last_valid_X,       # explain on held-out rows, never on training rows
            y=y, oof=oof,
            fold_scores=fold_scores,
            histories=histories,
            best_iters=best_iters,
            out_dir=PLOTS,
            shap_sample=shap_sample,
        )

    return cv


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--nrows", type=int, default=None,
                   help="limit rows per source table for a fast smoke test")
    p.add_argument("--folds", type=int, default=5,
                   help="number of CV folds")
    p.add_argument("--rounds", type=int, default=5000,
                   help="max boosting rounds; early stopping usually ends sooner")
    p.add_argument("--seed", type=int, default=42,
                   help="controls fold assignment; vary it to gauge CV stability")
    p.add_argument("--no-plots", action="store_true",
                   help="skip diagnostics (faster when iterating on features)")
    p.add_argument("--shap-sample", type=int, default=4000,
                   help="rows to explain with SHAP; lower is faster")
    p.add_argument("--use-tuned", action="store_true",
                   help="load hyperparameters from output/best_params.csv")
    a = p.parse_args()
    run(nrows=a.nrows, folds=a.folds, rounds=a.rounds, seed=a.seed,
        plots=not a.no_plots, shap_sample=a.shap_sample, use_tuned=a.use_tuned)
