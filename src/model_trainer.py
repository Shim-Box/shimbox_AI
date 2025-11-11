import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib 
import os

# data_processor에서 정의된 것과 동일한 Feature 목록 사용
FEATURES = [
    'skill', 'total_work_hours', 'delivery_count_yesterday', 'bmi', 'bmr', 
    'avg_heart_rate', 'steps', 'load_rel', 'strain', 'wish', 'driver_id',
    'time_per_delivery', 'deliveries_per_hour', 'steps_per_hour', 
    'steps_per_delivery', 'hr_per_step', 'hr_per_hour'
]
TARGET = 'theta_target'
DATA_PATH = 'data/processed/processed_logistics_data.csv'
MODEL_SAVE_PATH = 'models/optimal_capacity_predictor.pkl'

def train_and_save_model():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"❌ 오류: 데이터 파일 '{DATA_PATH}'을 찾을 수 없습니다. data_processor.py를 먼저 실행하세요.")
        return None

    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_features = ['skill', 'driver_id']
    numerical_features = [f for f in FEATURES if f not in categorical_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', 'passthrough', numerical_features)
        ],
        remainder='passthrough'
    )
    
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        tree_method='hist'
    )
    
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', xgb_model)])

    print("--- 모델 학습 시작 (theta_target 예측) ---")
    model_pipeline.fit(X_train, y_train)
    print("✅ 모델 학습 완료.")

    score = model_pipeline.score(X_test, y_test)
    print(f"테스트 데이터 R^2 Score (모델 정확도): {score:.4f}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model_pipeline, MODEL_SAVE_PATH)
    print(f"\n✅ 모델이 '{MODEL_SAVE_PATH}'에 저장되었습니다.")
    
    return model_pipeline

if __name__ == '__main__':
    train_and_save_model()