# src/ai/train_model.py
"""
daily_surveys.csv + daily_metrics.csv 로 내일 배송건수 예측 모델 학습

- surveys: date,courier_id,load_rel,strain,wish
- metrics: date,courier_id,work_hours,deliveries,bmi,avg_hr,steps
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

from src.config import (
    SURVEYS_CSV_PATH,
    METRICS_CSV_PATH,
    CAPACITY_MODEL_PATH,
    MODELS_DIR,
)


FEATURE_COLS = [
    "work_hours",
    "deliveries",
    "bmi",
    "avg_hr",
    "steps",
    "load_rel",
    "strain",
    "wish",
]


def build_training_dataframe() -> tuple[pd.DataFrame, pd.Series]:
    surveys = pd.read_csv(SURVEYS_CSV_PATH)
    metrics = pd.read_csv(METRICS_CSV_PATH)

    # date → datetime
    surveys["date"] = pd.to_datetime(surveys["date"])
    metrics["date"] = pd.to_datetime(metrics["date"])

    df = metrics.merge(surveys, on=["date", "courier_id"], how="inner")

    # courier_id + date 순으로 정렬
    df = df.sort_values(["courier_id", "date"])

    # 타겟: 다음날 deliveries
    df["next_deliveries"] = df.groupby("courier_id")["deliveries"].shift(-1)

    df = df.dropna(subset=["next_deliveries"])

    X = df[FEATURE_COLS].astype(float)
    y = df["next_deliveries"].astype(float)

    return X, y


def train_and_save_model():
    X, y = build_training_dataframe()

    print(f"[INFO] 학습 데이터 개수: {len(X)}")

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    bundle = {
        "model": model,
        "features": FEATURE_COLS,
    }
    joblib.dump(bundle, CAPACITY_MODEL_PATH)
    print(f"[OK] 모델 저장: {CAPACITY_MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
