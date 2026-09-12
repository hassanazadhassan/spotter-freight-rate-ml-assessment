from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from src.features import LaneEncoder, engineer, to_model_frame

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MODEL_PARAMS = dict(
    max_iter=800,
    max_depth=4,
    learning_rate=0.1,
    l2_regularization=0.5,
    min_samples_leaf=20,
    loss="absolute_error",
    early_stopping=True,
    n_iter_no_change=30,
    validation_fraction=0.1,
    categorical_features="from_dtype",
    random_state=42,
)

def eda_summary(raw_before_cleaning: pd.DataFrame, df: pd.DataFrame) -> None:
    print("\n--- EDA / data-quality summary ---")
    print(f"rows: {len(df):,}  date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print("missing values (raw):\n", raw_before_cleaning.isna().sum().loc[lambda s: s > 0])
    print(f"negative weight rows (raw, fixed via abs()): {(raw_before_cleaning['weight'] < 0).sum()}")
    rpm = df["posted_rate"] / df["distance"]
    print(f"loads priced >$5/mile (normal ~$2/mile): {(rpm > 5).sum()} "
          f"({(rpm > 5).mean() * 100:.2f}% of rows) -- spread evenly across months, "
          "likely spot-market premium/rush loads, not a data error. Left in training "
          "(loss='absolute_error' keeps them from destabilizing the model) rather than dropped.")
    print("posted_rate by equipment (mean):\n", df.groupby("equipment")["posted_rate"].mean().round(1))
    corr = df[["distance", "market_index", "quote_signal"]].apply(
        lambda s: s.corr(df["posted_rate"])
    )
    print("correlation with posted_rate:\n", corr.round(3))

def time_based_split(df: pd.DataFrame, holdout_days: int = 61):
    cutoff = pd.Timestamp(df["date"].max()) - pd.Timedelta(days=holdout_days)
    train = df[df["date"] <= cutoff].copy()
    valid = df[df["date"] > cutoff].copy()
    return train, valid

def fit_predict(train_raw: pd.DataFrame, valid_raw: pd.DataFrame):
    target_train = train_raw["posted_rate"]
    rpm_train = target_train / train_raw["distance"]

    encoder = LaneEncoder().fit(train_raw, rpm_train)
    X_train = to_model_frame(encoder.transform(train_raw))
    X_valid = to_model_frame(encoder.transform(valid_raw))

    model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    model.fit(X_train, target_train)
    preds = model.predict(X_valid)
    return model, encoder, preds

def evaluate(y_true, y_pred, label: str) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    print(f"{label}: MAE=${mae:,.2f}  RMSE=${rmse:,.2f}  MAPE={mape:.2f}%")

def main() -> None:
    raw_original = pd.read_csv(DATA / "train_test.csv")
    raw = engineer(raw_original)
    eda_summary(raw_original.assign(date=pd.to_datetime(raw_original["date"])), raw)

    raw_sorted = raw.sort_values("date")
    holdout_days = 61
    train_part, holdout_part = time_based_split(raw_sorted, holdout_days)
    print(f"\ntrain rows: {len(train_part):,}  holdout rows: {len(holdout_part):,}"
          f"  (holdout = last {holdout_days} days by date)")

    _, _, holdout_preds = fit_predict(train_part, holdout_part)
    print("\n--- Holdout validation (time-based split) ---")
    evaluate(holdout_part["posted_rate"], holdout_preds, "Full feature set")

    print("\n(ablation only affects this printed comparison, not the final model,")
    print(" since market_index/quote_signal are already excluded from FEATURE_COLUMNS)")

    print("\n--- Expanding-window cross-validation ---")
    dates_sorted = np.sort(raw_sorted["date"].unique())
    n_folds = 5
    fold_edges = np.array_split(dates_sorted, n_folds + 1)
    maes = []
    for i in range(1, n_folds + 1):
        cut = pd.Timestamp(fold_edges[i][0])
        fold_end = pd.Timestamp(fold_edges[i][-1]) + pd.Timedelta(days=1)
        fold_train = raw_sorted[raw_sorted["date"] < cut]
        fold_valid = raw_sorted[(raw_sorted["date"] >= cut) & (raw_sorted["date"] < fold_end)]
        if len(fold_train) < 500 or len(fold_valid) == 0:
            continue
        _, _, fold_preds = fit_predict(fold_train, fold_valid)
        fold_mae = mean_absolute_error(fold_valid["posted_rate"], fold_preds)
        maes.append(fold_mae)
        print(f"fold {i}: {len(fold_train):,} train / {len(fold_valid):,} valid -> MAE=${fold_mae:,.2f}")
    print(f"CV mean MAE: ${np.mean(maes):,.2f}  (+/- {np.std(maes):,.2f})")

    print("\n--- Refitting final model on full train_test.csv ---")
    target_full = raw_sorted["posted_rate"]
    rpm_full = target_full / raw_sorted["distance"]
    final_encoder = LaneEncoder().fit(raw_sorted, rpm_full)
    X_full = to_model_frame(final_encoder.transform(raw_sorted))
    final_model = HistGradientBoostingRegressor(**MODEL_PARAMS)
    final_model.fit(X_full, target_full)

    validation = pd.read_csv(DATA / "validation.csv")
    validation_eng = engineer(validation)
    X_validation = to_model_frame(final_encoder.transform(validation_eng))
    validation_preds = final_model.predict(X_validation)
    validation_preds = np.clip(validation_preds, 1.0, None)

    template = pd.read_csv(DATA / "validation_predictions_template.csv")
    out = template[["load_id"]].merge(
        pd.DataFrame({"load_id": validation["load_id"], "predicted_rate": validation_preds}),
        on="load_id",
        how="left",
    )
    out_path = ROOT / "validation_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(out):,} rows)")

    december = pd.read_csv(DATA / "december_chart_inputs.csv")
    december_eng = engineer(december.drop(columns=["predicted_rate"]))
    X_december = to_model_frame(final_encoder.transform(december_eng))
    december["predicted_rate"] = np.clip(final_model.predict(X_december), 1.0, None)
    dec_path = DATA / "december_chart_inputs.csv"
    december.to_csv(dec_path, index=False)
    print(f"wrote {dec_path} ({len(december)} rows)")
    print(december[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
