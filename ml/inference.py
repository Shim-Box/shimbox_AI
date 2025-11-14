import numpy as np
import joblib
import torch

from transformers import PatchTSTForRegression

from data_utils.feature_engineering import (
    encode_career,
    encode_condition,
    encode_finish,
)
from utils.logger import get_logger

logger = get_logger("inference")

RF_MODEL_PATH = "models/rf_capacity.pkl"
PATCHTST_MODEL_DIR = "models/patchtst_cap"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_rf_model = None
_patch_model = None


def load_rf():
    """RandomForest 모델 로드 (feature 이름은 코드에서 고정 사용)"""
    global _rf_model
    if _rf_model is None:
        _rf_model = joblib.load(RF_MODEL_PATH)
        logger.info(f"RF n_features_in_: {_rf_model.n_features_in_}")
    return _rf_model


def load_patchtst():
    global _patch_model
    if _patch_model is None:
        model = PatchTSTForRegression.from_pretrained(PATCHTST_MODEL_DIR)
        model.to(DEVICE)
        model.eval()
        _patch_model = model
    return _patch_model


def has_health_data(health: dict) -> bool:
    """실제 건강/설문 데이터 존재 여부"""
    if not health:
        return False
    keys = ["step", "heartRate", "finish1", "finish2", "finish3", "conditionStatus"]
    return any(k in health and health[k] is not None for k in keys)


# -------------------------------------------------------------------
# 현재 API 구조 기준: "오늘 하루 데이터"만 사용해서 예측
# -------------------------------------------------------------------
def build_today_features(profile_data: dict, health_data: dict, seq_len: int = 7):
    """
    profile_data: get_driver_profile()["data"]
    health_data: get_driver_health()["data"]
    """
    height = profile_data.get("height")
    weight = profile_data.get("weight")
    bmi = weight / ((height / 100) ** 2) if height and weight else 23.0

    career = profile_data.get("career", "experienced")

    work_minutes = profile_data.get("workDurationMinutes") or 0
    work_hours = work_minutes / 60 if work_minutes else 4.0

    steps = health_data.get("step", 8000)
    avg_hr = health_data.get("heartRate", 80)
    finish1 = health_data.get("finish1", "비슷했다")
    finish2 = health_data.get("finish2", "전혀 아니다")
    finish3 = health_data.get("finish3", "평소대로")
    condition = health_data.get("conditionStatus", "좋음")

    # RF에 사용할 feature 딕셔너리
    feat = {
        "bmi": bmi,
        "career_enc": encode_career(career),
        "finish1_enc": encode_finish(finish1),
        "finish2_enc": encode_finish(finish2),
        "finish3_enc": encode_finish(finish3),
        "condition_enc": encode_condition(condition),
        "steps_mean_7": float(steps),
        "work_hours_mean_7": float(work_hours),
        "deliveries_mean_7": 15.0,  # TODO: 실제 배송건수 API 나오면 교체
        "deliveries_std_7": 2.0,
    }

    # ✅ RandomForest에서 쓸 feature 순서를 코드에 고정 (10개)
    feature_names = [
        "bmi",
        "career_enc",
        "finish1_enc",
        "finish2_enc",
        "finish3_enc",
        "condition_enc",
        "steps_mean_7",
        "work_hours_mean_7",
        "deliveries_mean_7",
        "deliveries_std_7",
    ]

    x_rf_row = [feat[name] for name in feature_names]
    x_rf = np.array([x_rf_row], dtype="float32")

    # PatchTST input: (1, seq_len, num_channels)
    metrics = np.array(
        [[steps, avg_hr, work_hours, 15.0]], dtype="float32"
    )  # (1, 4)
    seq = np.repeat(metrics, seq_len, axis=0)  # (seq_len, 4)
    x_patch = torch.tensor(seq[None, ...], dtype=torch.float32)  # (1, seq_len, 4)

    return x_rf, x_patch, finish3


def predict_capacity(profile_data: dict, health_data: dict):
    # 직군 기반 기본값 (한글/영문 모두 대응)
    career_raw = profile_data.get("career", "experienced")

    base_cap_map = {
        # 한글
        "초보자": 10,
        "초보": 10,
        "경력자": 20,
        "경력": 20,
        "숙련자": 30,
        "숙련": 30,
        # 영문
        "beginner": 10,
        "experienced": 20,
        "expert": 30,
    }
    base_cap = base_cap_map.get(career_raw, 20)

    # 건강데이터/설문이 전혀 없으면 → 직군 기본값만 사용
    if not has_health_data(health_data):
        cap = apply_safety_clamp(base_cap, base_cap)
        logger.info(
            f"[NO_HEALTH] career={career_raw}, base_cap={base_cap} → final={cap}"
        )
        return cap

    # 건강데이터가 있을 때만 PatchTST + RF 사용
    seq_len = 7
    x_rf, x_patch, finish3 = build_today_features(profile_data, health_data, seq_len)

    # RF
    rf_model = load_rf()
    rf_cap = float(rf_model.predict(x_rf)[0])

    # PatchTST (HF)
    patch_model = load_patchtst()
    with torch.no_grad():
        x_patch = x_patch.to(DEVICE)
        outputs = patch_model(past_values=x_patch)
        patch_cap = float(outputs.regression_outputs[0, 0].cpu().item())

    pred_raw = min(base_cap, patch_cap, rf_cap)

    # TODO: 오늘 배송건수 API 나오면 교체
    today_deliveries = 10

    cap = apply_survey_rule(pred_raw, finish3, today_deliveries, base_cap)
    cap = apply_safety_clamp(cap, base_cap)

    logger.info(
        f"capacity 예측: base={base_cap:.1f}, patch={patch_cap:.1f}, rf={rf_cap:.1f}, final={cap}"
    )
    return cap

def apply_survey_rule(pred_cap_raw, pref, today_deliveries, base_cap):
    cap = pred_cap_raw

    if pref in ["적게", "less"]:
        # 오늘보다 최소 3개는 적게, 직군 기본값보다도 약간 낮게
        target = min(today_deliveries - 3, base_cap - 3)
        cap = min(cap, target)

    elif pref in ["더 많이", "more"]:
        # 살짝 늘려주기 (최대 직군 2배까지)
        cap = min(cap * 1.2, base_cap * 2)

    return cap

def apply_safety_clamp(cap, base_cap):
    cap = round(cap)
    cap = max(cap, 4)
    cap = min(cap, base_cap * 2)
    return cap
