"""Hyperparameter search for the Home Credit XGBoost model, with visual analysis.

WHY BAYESIAN SEARCH RATHER THAN GRID SEARCH
-------------------------------------------
A grid over 6 parameters at 4 values each is 4,096 fits; at several minutes per
fit that is impossible on a laptop. Optuna's TPE sampler instead models which
regions of the space produced good scores and concentrates its trials there, so
it typically finds a strong configuration in 30-60 trials.

WHAT THE PLOTS TELL YOU
-----------------------
Finding the best parameters is the smaller half of the value here. The bigger
half is learning WHICH parameters actually matter for this dataset, so you know
where to spend future effort:

  optimisation history  Has the search converged, or would more trials help?
  parameter importance  Which parameters drive the score at all? Usually only
                        2-3 of them do; the rest can be fixed at defaults.
  slice plot            For each parameter, what does the score-vs-value curve
                        look like? Reveals whether the best value sits at a
                        clear optimum or against a search-range boundary --
                        the latter means the range was set too narrow.
  parallel coordinates  Which COMBINATIONS work? Catches interactions that
                        per-parameter views miss.

    python src/tune.py --trials 40 --nrows 60000
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

import features

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "output")
PLOTS = os.path.join(OUT, "plots")

# Optuna logs every trial at INFO; quiet it so our own summary lines are visible.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X, y, folds, seed, rounds, early_stop):
    """One trial: sample a configuration, CV it, return mean validation AUC.

    SEARCH SPACE REASONING
      max_depth        3-9. Below 3 underfits badly; above 9 memorises given
                       ~1,600 features.
      min_child_weight 5-150, log scale. The main anti-overfit lever at an 8%
                       positive rate, and its effect is multiplicative, so log
                       spacing samples it far more sensibly than linear.
      eta              0.01-0.1, log. Interacts with tree count -- lower eta
                       needs more rounds, which early stopping supplies.
      subsample /      0.5-1.0. Below ~0.5 each tree sees too little to be
      colsample        useful.
      reg_alpha /      1e-3 to 10, log. Range spans four orders of magnitude
      reg_lambda       because the useful value is not known a priori.
    """
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "nthread": -1,
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "min_child_weight": trial.suggest_float("min_child_weight", 5, 150,
                                                log=True),
        "eta": trial.suggest_float("eta", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    iters = []  # collected per fold; the booster itself is freed each iteration

    for fold, (tr, va) in enumerate(skf.split(X, y)):
        dtr = xgb.DMatrix(X.iloc[tr], y[tr])
        dva = xgb.DMatrix(X.iloc[va], y[va])

        model = xgb.train(params, dtr, num_boost_round=rounds,
                          evals=[(dva, "valid")],
                          early_stopping_rounds=early_stop, verbose_eval=False)
        pred = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
        scores.append(roc_auc_score(y[va], pred))
        iters.append(int(model.best_iteration))
        del dtr, dva, model

        # PRUNING: report the running mean after each fold so Optuna can abort
        # clearly-losing configurations early instead of paying for all folds.
        # This is where most of the wall-clock saving in the search comes from.
        trial.report(float(np.mean(scores)), fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Store the tree count too: the best eta is meaningless without knowing how
    # many rounds it needed, and train.py has to reuse both together. Averaged
    # over folds, since each fold early-stops at a slightly different round.
    trial.set_user_attr("best_iteration", int(np.mean(iters)))
    return float(np.mean(scores))


def plot_study(study, out_dir):
    """Render Optuna's built-in analysis plots plus a custom history view."""
    os.makedirs(out_dir, exist_ok=True)
    from optuna import visualization as vis

    done = [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE]

    # --- Custom history: running best makes convergence obvious -------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    vals = [t.value for t in done]
    ax.scatter(range(len(vals)), vals, s=18, alpha=0.6, color="#0072B2",
               label="trial")
    ax.plot(range(len(vals)), np.maximum.accumulate(vals), color="#D55E00",
            lw=1.6, label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("mean CV AUC")
    ax.set_title("Optimisation history — flattening means converged")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(os.path.join(out_dir, "10_tuning_history.png"),
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir}/10_tuning_history.png")

    # --- Optuna's plotly figures, written as static HTML --------------------
    # Kept as HTML rather than PNG because they are interactive (hover shows
    # exact parameter values per trial) and need no extra image dependency.
    for name, fn in [
        ("11_param_importance", vis.plot_param_importances),
        ("12_slice", vis.plot_slice),
        ("13_parallel_coordinate", vis.plot_parallel_coordinate),
        ("14_contour", vis.plot_contour),
    ]:
        try:
            path = os.path.join(out_dir, f"{name}.html")
            fn(study).write_html(path)
            print(f"  wrote {path}")
        except Exception as e:
            print(f"  {name} skipped: {type(e).__name__}: {e}")


def run(trials=40, nrows=None, folds=3, rounds=3000, early_stop=100, seed=42):
    """Run the search, save the best parameters, and plot the analysis.

    Defaults trade fidelity for speed: 3 folds rather than 5, and typically a
    row subsample. The RANKING of configurations is stable under those
    reductions even though the absolute AUC is lower, which is all the search
    needs. Re-validate the winner with a full 5-fold train.py run.
    """
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    train, _, feats = features.build(DATA, nrows=nrows)
    X, y = train[feats], train["TARGET"].astype(int).values
    print(f"tuning on {X.shape[0]:,} rows x {X.shape[1]:,} features, "
          f"{trials} trials\n", flush=True)

    # MedianPruner stops a trial whose intermediate score is below the median of
    # completed trials at the same fold. n_warmup_steps=1 lets every trial finish
    # at least one fold, since a single fold is too noisy to judge on.
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
    )

    def cb(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"  trial {trial.number:3d}  auc={trial.value:.6f}  "
                  f"best={study.best_value:.6f}", flush=True)

    study.optimize(
        lambda t: objective(t, X, y, folds, seed, rounds, early_stop),
        n_trials=trials, callbacks=[cb], gc_after_trial=True)

    n_pruned = len([t for t in study.trials
                    if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\nbest AUC: {study.best_value:.6f}   "
          f"({len(study.trials)} trials, {n_pruned} pruned, "
          f"{time.time() - t0:.0f}s)")
    print("best params:")
    for k, v in study.best_params.items():
        print(f"  {k:20s} {v}")

    # Persist for reuse by train.py and for comparing successive searches.
    pd.DataFrame([study.best_params]).to_csv(
        os.path.join(OUT, "best_params.csv"), index=False)
    study.trials_dataframe().to_csv(
        os.path.join(OUT, "tuning_trials.csv"), index=False)

    plot_study(study, PLOTS)
    return study


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--nrows", type=int, default=None,
                   help="row cap per table; tuning on a subset is usually enough")
    p.add_argument("--folds", type=int, default=3,
                   help="fewer folds than final training, for speed")
    p.add_argument("--rounds", type=int, default=3000)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    run(trials=a.trials, nrows=a.nrows, folds=a.folds, rounds=a.rounds,
        seed=a.seed)
