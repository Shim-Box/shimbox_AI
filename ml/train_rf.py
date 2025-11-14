import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import pandas as pd

from data_utils.feature_engineering import build_rf_dataset
from utils.logger import get_logger

logger = get_logger("train_rf")

DATA_PATH = "data/train_history.csv"
MODEL_PATH = "models/rf_capacity.pkl"
FEATURE_NAMES_PATH = "models/rf_feature_names.txt"

def main():
    logger.info("RF 학습용 데이터 로드 중...")
    df = pd.read_csv(DATA_PATH)

    X, y, feature_names = build_rf_dataset(df)
    logger.info(f"X shape={X.shape}, y shape={y.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )

    logger.info("RandomForest 학습 시작...")
    model.fit(X_train, y_train)

    pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred)
    logger.info(f"✅ RF 학습 완료. MAE={mae:.3f}")

    joblib.dump(model, MODEL_PATH)
    logger.info(f"모델 저장: {MODEL_PATH}")

    # feature 이름도 같이 저장
    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
        for name in feature_names:
            f.write(name + "\n")
    logger.info(f"피처 이름 저장: {FEATURE_NAMES_PATH}")

if __name__ == "__main__":
    main()
