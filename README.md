**Salvador Cabeças, nº 20260659**
# Baseline Predictive Pipeline -- ETAI

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                  # entry point: run the whole pipeline
├── config.yaml               # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py               # loading
│   ├── preprocessing.py      # row-preserving cleaning (incl. domain-rule checks), training-only de-duplication, deployable preprocessing pipeline, and the split that locks the final test set away (week 4)
│   ├── model.py               # model construction (incl. the oblique random forest and AdaBoost builders, week 6) + build_pipeline(): preprocessing + model as one estimator (week 5)
│   ├── evaluate.py           # stratified k-fold cross-validation, out-of-fold report + fairness check (week 4), corrected paired test to compare two models (week 6), several metrics and out-of-fold probabilities (week 7)
│   ├── tuning.py             # hyperparameter tuning with Optuna + nested cross-validation (week 5)
│   └── results.py            # saves each run's report to disk, and its scores to the runs table (week 7)
├── results/                  # created automatically -- one report per run + runs.csv, the table of all runs (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md              # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check) and `src/preprocessing.py` (leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, **and** the train/test split itself, all in the one file rather than split across two) replace the old naive `dropna()`/`pd.get_dummies()` preprocessing; encoder/scaler pair (target encoding + standard scaling) chosen by an empirical grid over 15 repeated splits, checked against the runner-up with a paired comparison so the win isn't just noise; three redundant columns (found via correlation + VIF) dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections -- see "Preprocessing decisions" below. Threshold-independent metrics (ROC-AUC/PR-AUC) and a calibration check are deliberately **not** added yet -- not yet |
| 4 | Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly | A **locked final test set** (20%, stratified, seed 42) is set aside by `split_dev_test()` (replaces `split_train_test()`) and never scored; models are now judged by **stratified 5-fold cross-validation** of the whole pipeline (preprocessing + model) on the development set, reported per fold with mean ± std and the train-validation gap; the classification report and fairness check now use out-of-fold predictions; target encoding switched to scikit-learn's cross-fitting `TargetEncoder` (a row's own label never leaks into its own encoding), encoder/scaler set by hand in `config.yaml` (target encoding + robust scaling, reasons in the comments); **two fixes** in `clean_dataset()`: genuine `NaN`s in categorical columns were being turned into the string `"nan"` (a fake category), so 229 `c_charge_degree` gaps were never imputed or flagged -- fixed in `config.yaml` alone: `"nan"` added to `diagnostics.placeholder_tokens` (the category cleanup's last step turns listed tokens into `NaN`, after its text conversion); and it no longer drops rows -- de-duplication moved to a separate, training-only `drop_duplicate_rows()` (run before the dev/test split), so the same cleaning can run on new data where every row needs a prediction; `src/data_diagnostics.py` removed -- its one cleaning function (`flag_invalid_values`) moved into `preprocessing.py`, and the EDA-only checks (missingness test, duplicate counts) live in the EDA notebooks, not in every pipeline run; `dummy` (majority-class) model added as the floor to beat, and `random_forest` registered (sensible defaults, untuned); the final model is refit on the whole development set after CV; `config.yaml` gains `test_set` and `cv` sections -- see "Model evaluation" below |
| 5 | Hyperparameter tuning without leakage | New `src/tuning.py`: hyperparameters are tuned with **Optuna** (seeded TPE sampler) on the development set only -- every candidate is scored by stratified 5-fold CV of the **whole** pipeline (preprocessing re-fit in every fold of every trial), on the same folds for every candidate; the tuning procedure is estimated honestly by **nested cross-validation** (the tuning repeated inside each outer CV fold, scored on the outer fold it never saw), whose out-of-fold predictions now feed the classification report and fairness check; the reported "best trial" score is shown next to the nested one as an optimism check; the final model is the tuned pipeline refit on all development rows; `build_pipeline()` in `src/model.py` now builds the preprocessing + model `Pipeline` (one place to add a learned step later); `config.yaml` gains a `tuning` section (on/off, trials, inner folds, seed, one search space per model type) and the model becomes the decision tree; `optuna` added to `requirements.txt`. The locked test set is still untouched -- see "Hyperparameter tuning" below |
| 6 | Ensembles in depth: random forest, oblique random forest, AdaBoost | Two new model types in `src/model.py`: `oblique_random_forest` (a forest of oblique trees built from scikit-learn parts -- every tree splits on its own random sparse linear combinations of the features, via `SparseRandomProjection` inside `BaggingClassifier`) and `adaboost` (AdaBoost over decision trees of a chosen depth; stumps by default), both registered in `_MODELS`; the random forest is now the model `main.py` runs (200 trees, all CPU cores); `config.yaml` gains search spaces for the three ensembles -- the forests tune tree size (`max_depth`, `min_samples_leaf`) and decorrelation (`max_features` / number of combinations) but not the number of trees, AdaBoost tunes its base trees' depth, its number of rounds and its learning rate; `corrected_paired_test()` in `src/evaluate.py` compares two models scored on the same folds (Nadeau & Bengio's corrected paired t-test), so a "winner" is only called when the difference is bigger than fold-to-fold noise; `scipy` listed in `requirements.txt` (it already came with scikit-learn, so nothing new to install) -- see "Ensembles" below |
| 7 | Gradient boosting + evaluating with probabilities | Three new model types in `src/model.py`: `gradient_boosting` (scikit-learn's `GradientBoostingClassifier`: each shallow tree fits the gradient of the log-loss left by the trees before it), `hist_gradient_boosting` (the same idea on binned features: much faster on large data, and it handles missing values itself) and `naive_bayes` (`GaussianNB`, a fast probabilistic baseline); gradient boosting is now the model `main.py` runs, each with its search space in `config.yaml`; evaluation now reports **several metrics** (accuracy, balanced accuracy, precision, recall, F1, F-beta, ROC-AUC, PR-AUC) and one **selection metric** (`cv.selection_metric`, ROC-AUC) that tuning maximises and runs are ranked by; out-of-fold **probabilities** (not just labels) feed the reports, with a confusion matrix; `class_weight: balanced` in the commented examples of the models that support it; new `save_scores()` in `src/results.py`: every run adds a row to `results/runs.csv` (the model and its settings, the score on each fold, the no-model floor and the untuned score on the same folds), so any two runs on the same data and folds can be compared with `corrected_paired_test()`; the printed output was reorganised to be easier to read, with the full tables in the results file (optional: keep your own format if you prefer) -- see "Model evaluation" below |

## Preprocessing decisions

*(Written straight from the diagnosis in `Practical/W3/notebooks/01_eda_introduction.ipynb`; the preprocessing walkthrough is in `Practical/W4/notebooks/02_preprocessing.ipynb`. This is the summary.)*

| Column(s) | Issue found | Mechanism | What was done |
|---|---|---|---|
| `age` | 2.0% missing | MCAR | median impute, no indicator needed |
| `juv_fel_count` | 3.0% missing | MCAR | median impute, no indicator needed |
| `priors_count` | ~7% missing (incl. placeholder tokens) | MNAR -- tied to `age_cat` | median impute + `priors_count_was_missing` flag |
| `c_charge_degree` | 3.2% missing | MNAR -- tied to `age_cat` | mode impute + `c_charge_degree_was_missing` flag |
| `race` | ~1% missing (placeholder tokens) | MCAR | mode impute, no indicator (excluded from model features anyway) |
| `sex` | ~1.5% missing (incl. placeholder tokens) | MCAR | mode impute, no indicator needed |
| `age`, `decile_score`, `juv_fel_count`, `priors_count` | invalid values (out-of-range or negative) | domain rule | converted to `NaN` before imputation |
| `sex` / `race` / `c_charge_degree` / `score_text` | inconsistent category spelling (casing, whitespace, abbreviations) | data entry | canonicalized to one spelling per category |
| whole rows | 72 exact-duplicate rows, all sharing a repeated `id` | data entry | dropped, kept first occurrence |
| `prior_offenses`, `age_in_months`, `juvenile_total` | redundant with other columns (correlation r=1.00, or -- for `juvenile_total` -- an exact sum caught only by VIF) | multicollinearity | dropped |

**Encoder/scaler pair:** chosen by hand in `config.yaml` -- **target encoding** (compact, informative and **robust scaling** (median/IQR, so the few extreme counts don't set the scale). The alternatives (`onehot`/`ordinal`/`count`, `none`/`standard`/`minmax`) are one config change away.


## Hyperparameter tuning

*(Walkthrough: `Practical/W5/notebooks/04_hyperparameter_tuning.ipynb`.)* With `tuning.enabled: true` in `config.yaml`, `python main.py` also:

1. **Nested cross-validation** -- for each of the 5 outer CV folds, runs a full Optuna study on the outer-training rows only (each trial = 5-fold CV of the whole pipeline), refits the best candidate on those rows and scores it on the outer-validation fold. This is the honest estimate of the tuned model.
2. **Tuning on the whole development set** -- the same study, once, on every development row: these are the hyperparameters the final model uses.

The best trial's score is printed too, but it is **optimistic** (the maximum of many noisy CV scores) -- report the nested one. Search spaces live in `config.yaml` under `tuning.search_spaces`, one per model type, keyed by Pipeline parameter name (`model__max_depth`); a model without a search space can't be tuned until you add one. Cost: `(cv.n_splits + 1) x n_trials x tuning.n_splits` fits.

## Ensembles

Set `model.type` in `config.yaml` to choose the model; each type has a commented `params` example and its own search space under `tuning.search_spaces`.

| `model.type` | What it is | Search space (what the tuner chooses) |
|---|---|---|
| `random_forest` | many deep-ish trees on bootstrap samples, random features at every split, averaged | `max_depth`, `min_samples_leaf`, `max_features` |
| `oblique_random_forest` | the same, but each tree splits on random linear combinations of the features | the trees' `max_depth`, `min_samples_leaf`; `n_components` (combinations per tree) |
| `adaboost` | shallow trees built in sequence, each focused on the rows the previous ones got wrong; weighted vote | base-tree depth, `n_estimators`, `learning_rate` |

The forests' `n_estimators` is fixed in `params`, not tuned: more trees never overfit, they only cost time. Tuning an ensemble takes minutes per run (900 fits with the default settings) -- while testing, use a small `tuning.n_trials` or `tuning.enabled: false`.

## Model evaluation

**Metrics.** `cv.metrics` lists what every run reports; `cv.selection_metric` is the one that matters: tuning maximises it and the scoreboard ranks runs by it. The default is **ROC-AUC**: it scores how well the model *ranks* people by risk, without choosing a threshold, so COMPAS's own 1-10 risk score can be judged the same way. Accuracy, precision, recall and F1 depend on the threshold that turns a probability into a yes/no (0.5 in this pipeline). `fbeta_<beta>` is F-beta: `fbeta_2` counts recall twice as much as precision.

**The scoreboard.** Every `python main.py` adds one row to `results/runs.csv` (`save_scores()` in `src/results.py`): the run id, the data file, the model, whether it was tuned, the encoder and scaler, the model's params (and the tuned ones), the selection metric with its mean and std, the **untuned** score of the same model, the **no-model floor** (a dummy on the same folds) and the score on each fold (`fold_1` ... `fold_5`, from nested CV when tuning is on). The console prints the runs on the same data and metric, best first. The full report of each run is in `results/run_<run id>.txt`.

Two runs can only be compared if they used the **same data, cleaning and folds**: when you change the data file, the cleaning rules or `cv.random_state`, delete `results/runs.csv` and start a new table. (You could make this more automatic.) To compare two runs, from the project root:

```python
import pandas as pd
from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/runs.csv")
folds = [c for c in runs.columns if c.startswith("fold_")]
corrected_paired_test(runs.loc[0, folds], runs.loc[1, folds], test_train_ratio=1 / 4)   # rows 0 and 1; 5-fold: 1 / (5 - 1)
```

It returns the mean difference, the folds the first run wins, t and the p-value.

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run.

**When `requirements.txt` changes** (a week adds a package -- e.g. week 5 adds `optuna`), reinstall once, with the venv active: `pip install -r requirements.txt`. Otherwise `python main.py` fails with `ModuleNotFoundError`. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Environment Troubleshooting

Two Windows issues come up often enough to note here -- if you hit either, this saves you re-diagnosing it from scratch.

**PowerShell blocks the venv activation script, every new terminal.** The `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` line above only fixes it for that one terminal window -- close it and it's back. For a fix that actually sticks across sessions, run this **once**, instead:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
If it still doesn't stick (common on locked-down school/lab machines with a Group Policy that resets it on every logon), skip PowerShell entirely: use **Git Bash** (`source venv/Scripts/activate`) or **cmd.exe** (`venv\Scripts\activate.bat`) instead -- neither is affected by PowerShell's execution policy.

**Windows blocks the terminal/Python from reading or writing files in Documents (or Desktop/Pictures).** Shows up as an "Access is denied" error, or a silent failure to create/update a file, only when the project sits inside one of those folders. Two independent settings can cause this -- check both:
- **Windows Security -> Virus & threat protection -> Manage ransomware protection** -- turn off **Controlled folder access**, or add your terminal/Python/editor to its allowed-apps list.
- **Settings -> Privacy & security -> File system** -- make sure the terminal/Python has access.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, diagnoses and cleans the data (week 3), locks the final test set away, cross-validates preprocessing + model on the development set (week 4), tunes the model's hyperparameters if `tuning.enabled` is on (week 5 -- see "Hyperparameter tuning" above; minutes, not seconds, for the ensembles), and prints, in order:
- the model, the selection metric and the CV design; the size of the development set and of the locked test set (which is **not** used)
- **Default hyperparameters**: the cross-validation score on the selection metric, the train-validation gap (how much better the model looks on rows it trained on: a large gap means it memorised) and the no-model floor
- with tuning on, **Tuning**: the nested cross-validation score (the honest score of the tuned model) and the chosen hyperparameters
- **All metrics**: mean ± std of every metric
- **Scoreboard**: this run next to the earlier runs on the same data and metric, best first
- the final model (refit on all development rows) and where the full report was saved

The full detail -- per-fold tables for every metric, the tuning report, the out-of-fold confusion matrix and classification report, the false-positive-rate-by-race comparison with COMPAS's own score -- goes to the results file.

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Push to GitHub via Terminal

Standard workflow, from the project's root folder, with the venv active:
```bash
git add .
git commit -m "short description of what changed"
git push
```

**If `git push` asks for a password and rejects your normal GitHub password:** GitHub no longer accepts account passwords for git over HTTPS -- you need a **Personal Access Token (PAT)** instead.
1. On GitHub: **Settings -> Developer settings -> Personal access tokens -> Tokens (classic)** -> **Generate new token**, with at least `repo` scope.
2. When `git push` prompts for a password, paste the token instead (username stays your GitHub username).
3. So you're not asked every time: `git config --global credential.helper manager` (Windows, usually already set up by Git for Windows) or `git config --global credential.helper store` (caches it in plaintext -- fine on a personal machine, not a shared one).

Alternative: set up an SSH key once (`ssh-keygen -t ed25519`, then add the public key under **GitHub -> Settings -> SSH and GPG keys**) and use the repo's SSH remote URL (`git@github.com:...`) instead of HTTPS -- no token to manage or renew.

## Dataset

See `data/README.md`.

## My experiments (Salvador)

### Week 4: preprocessing and cross-validation

| Model | Encoder | CV validation (mean ± std) | Gap | FPR African-American / Caucasian |
|---|---|---|---|---|
| Logistic regression | target | 0.672 ± 0.013 | +0.003 | 0.26 / 0.13 |
| Logistic regression | onehot | 0.670 ± 0.016 | +0.002 | 0.30 / 0.16 |
| Decision tree | target | 0.611 ± 0.015 | +0.084 | 0.35 / 0.25 |

Week 3 (single split, decision tree): test accuracy 0.657.

- One change tested: target encoder vs one-hot. The accuracy difference (0.002) is well below one standard deviation, so it is noise. One-hot raised false positives for every group, so the target encoder stays.
- CV changes the picture: the week 3 score of 0.657 came from one optimistic split (with older preprocessing). Under CV the tree scores 0.611 and overfits (gap +0.084). Logistic regression beats it by about 4 standard deviations, so it becomes the model.
- Removing race from the features does not remove the disparity: African-Americans still get about twice the false positive rate of Caucasians.

### Weeks 5-6: hyperparameter tuning and ensembles

All five models tuned with Optuna (30 trials, 5 inner folds) and scored with nested CV on the same 5 outer folds. For logistic regression I added a search space for `C` (log scale, 0.0001 to 100), the week 5 challenge. The test set is still untouched.

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 (+0.003) | 0.673 ± 0.016 (+0.004) | 0.678 | +0.002 | C=0.0715 |
| Decision tree | 0.611 ± 0.015 (+0.084) | 0.675 ± 0.020 (+0.009) | 0.680 | -0.000 | max_depth=9, min_samples_leaf=101, criterion=gini |
| Random forest | 0.652 ± 0.019 (+0.081) | 0.678 ± 0.014 (+0.011) | 0.683 | +0.004 | max_depth=22, min_samples_leaf=42, max_features=0.5 |
| Oblique random forest | 0.650 ± 0.009 (+0.113) | 0.682 ± 0.014 (+0.011) | 0.683 | -0.000 | max_depth=4, min_samples_leaf=2, n_components=10 |
| AdaBoost | 0.676 ± 0.012 (+0.006) | 0.682 ± 0.016 (+0.005) | 0.684 | -0.000 | max_depth=4, n_estimators=220, learning_rate=0.0251 |

- Tuning helped the models that were overfitting. The tree went from 0.611 to 0.675 and its gap dropped from +0.084 to +0.009, and both forests gained about 3 points. AdaBoost and logistic regression barely moved because they weren't overfitting in the first place. For logistic regression the gain is basically zero: the outer folds picked C anywhere between 0.035 and 1.77 and the score stayed the same. A linear model on 10 features doesn't have much variance for regularisation to remove.
- The chosen hyperparameters fit how each model works. The forest keeps deep trees (max_depth=22) but needs at least 42 people per leaf, so it stops memorising single cases and lets the averaging deal with the rest. AdaBoost goes the other way: fairly weak trees (depth 4), a small learning rate and more rounds.
- No model really wins. AdaBoost has the best mean (0.682), but the corrected paired t-test gives p between 0.26 and 0.86 against every other model, and against the oblique forest it only wins 2 of the 5 folds. Everything ends around 0.68, so the limit looks like the features, not the model. I keep the random forest: it's tied with the best, and it's the model in the upstream config, so my config stops conflicting on every merge.
- The number I report is the nested score (random forest 0.678 ± 0.014), not the best trial (0.683). The best trial is the maximum of 30 noisy CV scores, so part of it is luck. The optimism is small on average (+0.004 for the forest), but per fold it goes from -0.013 to +0.031, so with 5 folds that average is mostly noise. Fold 2 is the hard one for every model (about 0.65-0.66) and it's also where the tuning was most optimistic.
- Fairness looks like week 4. Every model gives African-Americans about twice the false positive rate of Caucasians (random forest 0.30 vs 0.17, COMPAS 0.45 vs 0.23). Logistic regression has the lowest rates (0.25 vs 0.12), but the ratio is still about 2x, so it seems to predict "reoffends" less often overall rather than being fairer.

### Week 7: boosting and the scoreboard

Every model below was tuned with Optuna (30 trials, 5 inner folds) and scored with nested CV on the same 5 outer folds, one run per model in `results/runs.csv`. Logistic regression and the dummy have no search space, so they only have the untuned score. The test set is still untouched.

**Selection metric: ROC-AUC.** COMPAS is used to rank people by risk, and ROC-AUC measures exactly that ranking (how often someone who reoffended gets a higher score than someone who didn't), without depending on the 0.5 cut-off. Where to put the threshold is a separate decision, and here it's closer to a policy choice than a modelling one, so I don't want the metric to fix it for me. It's also less noisy than accuracy when comparing models fold by fold, which was the problem last week.

| Model | Untuned | Tuned (nested CV) | Floor (dummy) |
|---|---|---|---|
| Gradient boosting | 0.727 | 0.728 ± 0.014 | 0.500 |
| HistGradientBoosting | 0.717 | 0.728 ± 0.015 | 0.500 |
| Random forest | 0.684 | 0.727 ± 0.016 | 0.500 |
| Oblique random forest | 0.686 | 0.727 ± 0.015 | 0.500 |
| AdaBoost | 0.723 | 0.727 ± 0.014 | 0.500 |
| Logistic regression | 0.718 | not tuned | 0.500 |
| Decision tree | 0.612 | 0.716 ± 0.015 | 0.500 |
| Naive Bayes | 0.685 | 0.705 ± 0.012 | 0.500 |
| Dummy | 0.500 | not tuned | 0.500 |

Best model (gradient boosting) against each of the others, corrected paired t-test on the 5 outer folds:

| Gradient boosting vs | Mean diff | Folds won | p |
|---|---|---|---|
| HistGradientBoosting | +0.000 | 3/5 | 0.976 |
| Random forest | +0.001 | 3/5 | 0.561 |
| Oblique random forest | +0.002 | 4/5 | 0.405 |
| AdaBoost | +0.002 | 4/5 | 0.336 |
| Logistic regression | +0.010 | 5/5 | 0.035 |
| Decision tree | +0.013 | 5/5 | 0.027 |
| Naive Bayes | +0.023 | 5/5 | 0.004 |

Tuning mattered most for the models that were overfitting: the tree went from 0.612 to 0.716 and both forests gained about 4 points, while gradient boosting barely moved (0.727 to 0.728), so its default settings were already close to what this data needs. After tuning, the five ensembles end up within 0.002 of each other, and against the other four the test gives p between 0.34 and 0.98, so gradient boosting has the best mean but there's no real winner among them.

What changes with ROC-AUC is that the test can now separate the best model from the simpler ones. Gradient boosting beats logistic regression, the tuned tree and Naive Bayes in all 5 folds, with p below 0.05, while last week, with accuracy, the best model couldn't be separated from any other (p between 0.26 and 0.86). The gain over logistic regression is still small (+0.010), though. Naive Bayes is the weakest real model: even tuned it flags very few people (recall 0.25, precision 0.74), probably because it treats correlated features like `age` and `age_cat` as independent.

`config.yaml` now runs gradient boosting, as the week 7 notebook asks. It's also the top of my scoreboard, but given the tests above I'd treat it as tied with the other ensembles.
