# Freight Rate Prediction

Predicts freight `posted_rate` from load features (distance, weight, equipment,
lane, date) using `HistGradientBoostingRegressor`, validated with a time-based
split that mirrors the real Nov/Dec forecast task.

## Repo layout
```
data/                    train_test.csv, validation.csv, template, december inputs
src/features.py          cleaning + feature engineering + LaneEncoder
src/train.py             EDA -> time-based holdout -> CV -> final fit -> writes predictions
score.py                 instructor-provided validator + December chart generator
colab_run.py             single-file version for Google Colab (no repo needed)
validation_predictions.csv
scorer_results/candidate_december.png
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```
`src.train` writes `validation_predictions.csv` (12,000 rows) and fills in
`data/december_chart_inputs.csv`. `score.py` checks both files are formatted
correctly and renders `scorer_results/candidate_december.png`.

## Run on Google Colab
See `colab_run.py` — paste its contents into a Colab cell after uploading the
4 data files. No local Python install needed.

## Approach summary
- **Split**: time-based, not random K-fold. Train covers Jan-Oct 2025;
  `validation.csv` covers Nov-Dec 2025 - dates the model never saw. A random
  split would leak future information into training, so the holdout is the
  trailing 61 days of `train_test.csv`, and CV uses 5 expanding-window folds
  sorted by date.
- **Data quality**: 292 negative `weight` values (sign errors, fixed with
  `abs()`), 300 missing `weight` (imputed by equipment-type median), 374
  missing `market_index`. No duplicate `load_id`s. All 64 pickup/delivery
  cities are shared across train, validation, and the December file, so
  city-level historical encodings generalize to unseen lanes.
- **Outliers**: 317 loads (0.66%) priced 3-7x the normal rate-per-mile, spread
  evenly across all 10 months - read as real spot-market/rush premiums, not
  data errors, so they're kept in training rather than dropped.
- **Features excluded on purpose**: `market_index` and `quote_signal`
  correlate with `posted_rate` at only ~0.03 / -0.04, and don't exist at all in
  `december_chart_inputs.csv`. Dropping both lets one model serve validation
  and December predictions identically.
- **Features used**: distance, weight (+ log/ratio transforms), calendar
  features (month, day-of-week, day-of-year, cyclical sin/cos encodings,
  weekend/month-edge flags), and smoothed historical rate-per-mile + frequency
  encodings for pickup, delivery, and equipment (fit on training data only).
- **Model**: `HistGradientBoostingRegressor` - native categorical support for
  pickup/delivery/equipment, no leakage-prone one-hot of 64x64 lane pairs.
- **Hyperparameters**: tuned via a 32-combo grid over
  max_depth/learning_rate/l2_regularization/min_samples_leaf/loss, scored on
  the same 5-fold expanding-window CV used for validation. Switching `loss`
  from `squared_error` to `absolute_error` was the dominant win - it cut mean
  CV MAE from ~$190 to ~$121 and cut fold-to-fold std from ~$42 to ~$14, since
  the outlier loads above were destabilizing squared-error loss. Final config:
  `max_depth=4, learning_rate=0.1, l2_regularization=0.5, min_samples_leaf=20,
  loss=absolute_error, max_iter=800, early_stopping=True`.

## Results

| Metric | Before tuning | After tuning |
|---|---|---|
| Holdout MAE | $139.79 | **$110.76** |
| Holdout MAPE | 6.62% | **4.88%** |
| CV mean MAE (5 folds) | $189.85 | **$120.82** |
| CV std across folds | ±$41.54 | **±$13.87** |
