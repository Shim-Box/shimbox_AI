SCALER_MIN = 0.0
SCALER_RANGE = 100.0
_patchtst_model = None

def load_patchtst_model():
    global _patchtst_model
    _patchtst_model = "PatchTST_Model_Loaded"
    print("✅ PatchTST 모델 로드 완료(더미)")

def patchtst_predict(X_seq):
    # 0~1 사이 예측값 가정 (더미)
    return 0.42
