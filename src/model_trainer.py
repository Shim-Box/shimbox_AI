import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from pathlib import Path
import joblib

# 고정 Feature 목록
FEATURES = [
    "skill", "total_work_hours", "delivery_count_yesterday", "bmi", "bmr",
    "avg_heart_rate", "steps", "load_rel", "strain", "wish", "driver_id",
    "time_per_delivery", "deliveries_per_hour", "steps_per_hour",
    "steps_per_delivery", "hr_per_step", "hr_per_hour",
]
TARGET = "theta_target"

DATA_PATH = Path("data/processed/processed_logistics_data.csv")
MODEL_SAVE_PATH = Path("models/optimal_capacity_predictor.pkl")


def train_and_save_model():
    if not DATA_PATH.exists():
        print(f"❌ 오류: 데이터 파일 '{DATA_PATH}'을 찾을 수 없습니다.")
        print("➡ 먼저 data_processor.py를 실행해 전처리 데이터를 생성하세요.")
        return None

    print("\n--- 데이터 로드 중 ---")
    df = pd.read_csv(DATA_PATH)

    # 결측치 처리
    if df.isnull().any().any():
        print("⚠️ 결측치가 발견되어 0 또는 'unknown'으로 대체합니다.")
        df.fillna({"skill": "unknown", "driver_id": "unknown"}, inplace=True)
        df.fillna(0, inplace=True)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_features = ["skill", "driver_id"]
    numerical_features = [f for f in FEATURES if f not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("num", "passthrough", numerical_features),
        ],
        remainder="drop",
    )

    xgb_model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        tree_method="hist",
    )

    model_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", xgb_model)]
    )

    print("\n--- 모델 학습 시작 (theta_target 예측) ---")
    model_pipeline.fit(X_train, y_train)
    print("✅ 모델 학습 완료.")

    # 평가
    y_pred = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 평가 결과:")
    print(f"   • R² Score : {r2:.4f}")
    print(f"   • MAE      : {mae:.4f}")

    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_SAVE_PATH)
    print(f"\n💾 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다.\n")

    return model_pipeline


if __name__ == "__main__":
    train_and_save_model()
