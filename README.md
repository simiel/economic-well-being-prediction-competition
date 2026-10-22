# Economic Well-Being Prediction

This repository is Samuel Mensah's modeling work for the Zindi Economic Well-Being Prediction Challenge, a beginner practice competition. The task is to predict a wealth measure (`Target`) for locations across Africa from geospatial covariates: built-up land and population density, land-cover fractions, nighttime lights, and distance to the capital and the shoreline. Training covers 21,454 locations in 18 countries (1994–2016). The 7,194 test locations sit in seven other countries (1998–2016), so a useful model has to score places in countries that never appear in training. The repository holds the challenge CSVs, the exploration and modeling notebooks, and the submission files those runs produced.

## Key features

- Exploratory checks on the training table: schema, summary statistics, the target distribution, feature variance, and the urban/rural and nighttime-lights relationships.
- A scikit-learn random-forest baseline on the numeric covariates, plus a binary urban/rural indicator.
- A CatBoost regressor, and a weighted blend of its test predictions with the random forest (20% CatBoost, 80% random forest).
- A CatBoost `GridSearchCV` over `colsample_bylevel`, `depth`, `boosting_type`, and `bootstrap_type`.
- A LightGBM pipeline in `solutionX.ipynb`: frequency encodings, aggregate land features, 10-fold cross-validation, early stopping, and predictions clipped to a fixed range before export.
- A trail of `ID` / `Target` submission CSVs, including Colab exports whose filenames mark later XGBoost and blend runs.

## Problem and data

Each row is one location. `Train.csv` has 19 columns and `Test.csv` has the same columns without `Target`. `SampleSubmission.csv` is the test `ID`s with a placeholder `Target`.

Inputs shipped with the challenge:

- `country`, `year`, and `urban_or_rural` (`R` or `U`)
- GHSL fields: water surface, built-up fractions for four periods (`pre_1975`, `1975_to_1990`, `1990_to_2000`, `2000_to_2014`), non-built-up fraction, and population density
- Land cover: crops, urban, and permanent and seasonal water within 10 km
- `nighttime_lights`, `dist_to_capital`, and `dist_to_shoreline`

Training countries, as stored in `Train.csv`: Cameroon, Central African Republic, `Cote d'Ivoire`, Ethiopia, Ghana, Guinea, Kenya, Lesotho, Malawi, Mali, Mozambique, Nigeria, Rwanda, Senegal, Sierra Leone, Swaziland, Tanzania, and Togo.

Test countries: Angola, Benin, Burkina Faso, Democratic Republic of Congo, Uganda, Zambia, and Zimbabwe.

The two country sets are disjoint. The starter notebook states the consequence: a random row split keeps the same countries on both sides and is an easier test than the leaderboard, which scores unseen countries. The training table has no missing values. `Target` runs from 0 to 1, with a mean of 0.351. Of the training locations, 14,061 are rural and 7,393 are urban. The same urban/rural imbalance appears in the test set (4,613 rural, 2,581 urban).

The metric used in the notebooks, and the metric the starter notebook identifies as the Zindi score, is root mean squared error.

## Approach

**Baseline.** `Starter_Notebook.ipynb` models the 14 numeric covariates from `ghsl_water_surface` through `dist_to_shoreline`. It holds out 20% of rows (`train_test_split`, `random_state=58`) and fits `RandomForestRegressor`. The first engineered feature is `is_urban`, set to 1 when `urban_or_rural` is `U`, and it is added to both train and test. The same notebook then fits a default `CatBoostRegressor` (1,000 iterations) and blends the two test predictions:

```python
blend = catp * 0.2 + randp * 0.8
```

That blend is written to `second_submission4.csv`.

**CatBoost search.** `Starter_Notebook copy.ipynb` continues from that setup and runs `GridSearchCV` with 5-fold cross-validation and `refit=True`. The grid is `colsample_bylevel` in `{0.01, 0.1}`, `depth` in `{1, 12}`, `boosting_type` in `{Ordered, Plain}`, and `bootstrap_type` in `{Bayesian, Bernoulli, MVS}`. The notebook stores the training log and writes `second_submission5_grid.csv`. It does not print `best_params_` or the best cross-validation score.

**LightGBM.** `solutionX.ipynb` concatenates train and test, then adds:

- frequency counts of `country`, `year`, and `urban_or_rural` (`count_country`, `count_year`, `count_urban_or_rural`)
- `all_ghsl`, the sum of the four built-up-period fractions
- `all_landcover_fraction`, crops plus urban land cover
- `all_waters`, permanent plus seasonal water within 10 km

The model matrix drops `ID`, raw `country`, raw `urban_or_rural`, raw `year`, and `Target`, leaving 20 features. `LGBMRegressor` uses gradient boosting (`gbdt`), learning rate 0.05, max depth 4, 150 leaves, `min_data_in_leaf` 30, and L2 regularization `reg_lambda` 75, for up to 1,500 rounds with early stopping after 50 rounds without improvement on the fold validation set. Validation is shuffled `KFold` with 10 splits and `random_state=2021`. Test predictions are the mean of the fold models, clipped to `[0.141, 0.808657]`, and saved as `XXXX.csv`.

`GroupKFold` is imported in `solutionX.ipynb`. The run that printed the saved scores uses shuffled `KFold`, so those scores come from a random row split. The starter notebook is where the country shift is called out, and it leaves a country-wise evaluation as an exercise.

## Results recorded in the repo

No Zindi leaderboard score is stored in the repository. The figures below are the metrics printed by the notebooks, or written by CatBoost's own training log.

| Run | How it was scored | Result |
| --- | --- | --- |
| Random forest, 14 numeric features | 20% random holdout | R² 0.6979, RMSE 0.1083 |
| Random forest plus `is_urban` | Same split | RMSE 0.1046 |
| LightGBM, engineered features | 10-fold shuffled KFold, out of fold | RMSE 0.0863 |

LightGBM fold RMSEs run from 0.0827 (fold 3) to 0.0925 (fold 2). The out-of-fold figure printed after the ten folds is 0.0863.

The default CatBoost fit logs training RMSE in `catboost_info/learn_error.tsv`: it falls from 0.1859 to 0.0837 over 1,000 iterations. That series is loss on the rows the model was fitting, and the notebook does not record a CatBoost holdout RMSE. The grid search was executed; its chosen parameters and cross-validation score were not written out.

## Tech stack

- Python notebooks (nbformat 4; `Notebooks/Explore_data.ipynb` records a Python 3.8 kernel)
- pandas and NumPy for the tables
- Matplotlib, plus pandas plots. The exploration notebook also imports seaborn; the saved plots are a target histogram, an urban/rural box plot, a nighttime-lights scatter, and a predicted-versus-actual scatter
- scikit-learn: `RandomForestRegressor`, `train_test_split`, `GridSearchCV`, `KFold`, and `mean_squared_error`
- CatBoost (`CatBoostRegressor`) and LightGBM (`LGBMRegressor`)
- Jupyter locally. `solutionX.ipynb` also contains commented Google Drive copies for a Colab session, and `colab/` stores prediction files exported from those runs

## Repository layout

```
Train.csv, Test.csv, SampleSubmission.csv   Challenge data
Notebooks/Explore_data.ipynb                 Schema, describe, target histogram, variance
Notebooks/Starter_Notebook.ipynb             Zindi starter, through the first CatBoost exercise
Starter_Notebook.ipynb                       Baseline, urban feature, CatBoost, 0.2/0.8 blend
Starter_Notebook copy.ipynb                  CatBoost grid search on that same setup
solutionX.ipynb                              LightGBM features, 10-fold training, clipped export
catboost_info/                               Training log from the default CatBoost fit
colab/                                       Prediction CSVs exported from Colab
*.csv at the repository root                 Submission files from the local runs
```

`Notebooks/Starter_Notebook.ipynb` writes `first_submission.csv`. The later root CSVs (`first_submission_RU.csv`, `second_submission2.csv`, `second_submission3.csv`, `second_submission5_grid1.csv`, `second_submission5_gridblend.csv`, `YYYY.csv`, `blend2.csv`, `catp_1col.csv`, and `Notebooks/second_submission.csv`) are further `ID` / `Target` submissions from the same effort. `colab/` contains prediction files only. Names such as `colab12_xg.csv`, `xg_pre.csv`, and `colab18blend.csv` mark XGBoost and blend exports. The notebooks that trained those Colab files are not in this repository.

## Getting started

Install pandas, NumPy, Matplotlib, seaborn, scikit-learn, CatBoost, and LightGBM in the environment you use to open the notebooks. The repository ships no dependency manifest, environment file, test suite, or build script, and the notebooks read no environment variables. `Train.csv`, `Test.csv`, and `SampleSubmission.csv` are already at the repository root. The notebooks are the entry points.

Open them with the repository root as the working directory so the root notebooks resolve `Train.csv` and `Test.csv`:

- `Notebooks/Explore_data.ipynb` reads `../Train.csv`, `../Test.csv`, and `../SampleSubmission.csv`.
- `Starter_Notebook.ipynb` trains the random forest and the CatBoost blend, then writes `second_submission4.csv`.
- `Starter_Notebook copy.ipynb` runs the CatBoost grid search and writes `second_submission5_grid.csv`. That search fits a CatBoost model for every grid point and every fold, so it is the long run in the set.
- `solutionX.ipynb` trains the LightGBM folds and writes `XXXX.csv`.

Re-running a notebook overwrites the submission file that notebook saves.

## Project status

This is a finished practice-challenge archive. Commits run from 12 March 2022 through 22 March 2022 under the author name Simiel, from the initial "Zindi Beginner Practice Challenge" commit through the notebook iterations and submission files. The repository is the notebook record of that work.

## What this demonstrates

The work is a complete pass through a tabular prediction problem whose real difficulty is geographic shift. It shows how to read a competition table, score models with the official RMSE, and recognize when a random split is easier than the test distribution the challenge actually uses. On top of the starter baseline it shows the modeling steps that move the local error: a numeric urban/rural flag, aggregate settlement, land-cover, and water features, frequency counts in place of raw categories, LightGBM trained out of fold with early stopping, a CatBoost hyperparameter search, and a weighted blend of two models. The random-forest holdout and the LightGBM out-of-fold run are both reported as RMSE. The saved submission files keep the intermediate attempts alongside the strongest recorded holdout or out-of-fold score, the LightGBM out-of-fold RMSE of 0.0863.
