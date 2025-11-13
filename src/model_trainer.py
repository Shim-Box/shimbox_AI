# src/model_trainer.py
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ====== 경로 안전화 ======
# project-root/
#   ├─ data/processed/processed_logistics_data.csv
#   └─ models/optimal_capacity_predictor.pkl
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "processed_logistics_data.csv"
MODEL_SAVE_PATH = ROOT / "models" / "optimal_capacity_predictor.pkl"

# ====== 고정 Feature / Target ======
FEATURES: List[str] = [
    "skill", "total_work_hours", "delivery_count_yesterday", "bmi", "bmr",
    "avg_heart_rate", "steps", "load_rel", "strain", "wish", "driver_id",
    "time_per_delivery", "deliveries_per_hour", "steps_per_hour",
    "steps_per_delivery", "hr_per_step", "hr_per_hour",
]
TARGET = "theta_target"


def _make_ohe():
    """sklearn 버전별 OneHotEncoder 호환 처리."""
    from sklearn.preprocessing import OneHotEncoder
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """지정한 컬럼을 숫자형으로 안전 변환 (결측은 0 대체)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def train_and_save_model():
    if not DATA_PATH.exists():
        print(f"❌ 오류: 데이터 파일 '{DATA_PATH}'을 찾을 수 없습니다.")
        print("➡ 먼저 전처리 파이프라인을 실행해 processed CSV를 생성하세요.")
        return None

    print("\n--- 데이터 로드 ---")
    df = pd.read_csv(DATA_PATH)

    # 컬럼 보정/점검

    if "driver_id" not in df.columns and "courier_id" in df.columns:
        df["driver_id"] = df["courier_id"].astype(str)

    # 누락 컬럼 체크
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"데이터에 다음 컬럼이 없습니다: {missing}")

    # 결측치 1차 처리 (범주형/식별자)
    df["skill"] = df["skill"].fillna("unknown").astype(str)
    df["driver_id"] = df["driver_id"].fillna("unknown").astype(str)

    # 숫자형 후보
    categorical_features = ["skill", "driver_id"]
    numerical_features = [f for f in FEATURES if f not in categorical_features]

    # 숫자형 변환 + 결측 보정
    df = _coerce_numeric(df, numerical_features + [TARGET]).fillna(0)

    # Train/Test Split
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 전처리/모델 파이프라인
    ohe = _make_ohe()
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", ohe, categorical_features),
            ("num", "passthrough", numerical_features),
        ],
        remainder="drop",
    )

    # XGBoost 가져오기
    try:
        from xgboost import XGBRegressor
    except Exception as e:
        raise RuntimeError(
            "xgboost가 설치되어 있지 않습니다. requirements.txt에 'xgboost'를 추가하고 설치하세요."
        ) from e

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

    print("\n📊 평가 결과")
    print(f"   • R² Score : {r2:.4f}")
    print(f"   • MAE      : {mae:.4f}")

    # 저장
    MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_SAVE_PATH)
    print(f"\n💾 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다.\n")

    return model_pipeline


if __name__ == "__main__":
    train_and_save_model()
