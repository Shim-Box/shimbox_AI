# src/config.py
from pathlib import Path
import os

# ===========================
#  API / 인증 설정
# ===========================
API_BASE_URL = "http://116.39.208.72:26443"

# 관리자 계정 (ENV 우선, 없으면 기본)
ADMIN_EMAIL = os.getenv("SB_ADMIN_EMAIL", "admin@gmail.com")
ADMIN_PASSWORD = os.getenv("SB_ADMIN_PASS", "12341234")

# ===========================
#  데이터 경로
# ===========================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

SURVEYS_CSV_PATH = BASE_DIR / "data" / "raw" / "daily_surveys.csv"
METRICS_CSV_PATH = BASE_DIR / "data" / "raw" / "daily_metrics.csv"

PRODUCT_EXCEL_PATH = DATA_DIR / "raw" / "상품 더미.xlsx"

CAPACITY_MODEL_PATH = MODELS_DIR / "capacity_model.pkl"

# ===========================
#  비즈니스 로직 설정
# ===========================
# 행정구 필터 (주소에 이 문자열이 들어간 상품만 배정)
DISTRICT_FILTER = "성북구"

# 직군별 기본 물량
ROLE_BASE_CAP = {
    "초보자": 10,
    "경력자": 20,
    "숙련자": 30,
}

# finish3 → wish (1/2/3) 맵핑
FINISH3_TO_WISH = {
    "더 적게": 1,
    "적게": 1,
    "조금 적게": 1,
    "평소대로": 2,
    "보통": 2,
    "그냥 그래요": 2,
    "더 많이": 3,
    "많이": 3,
}

# wish(1/2/3)에 따른 가중치
# 10 → (finish3=적게) 6 / (평소) 10 / (더 많이) 16 정도 느낌
WISH_FACTOR = {
    1: 0.6,
    2: 1.0,
    3: 1.6,
}

# ShimBox driverId → CSV courier_id 매핑
# 예시: driverId=1 → "C_01"
DRIVER_ID_TO_COURIER_ID = {
    # 필요시 여기에 직접 매핑 추가
    # 1: "C_01",
    # 2: "C_02",
}


def normalize_career(raw: str | None) -> str:
    if not raw:
        return "경력자"
    s = str(raw).strip().lower()
    if any(k in s for k in ["초보", "신입", "beginner", "junior"]):
        return "초보자"
    if any(k in s for k in ["숙련", "senior", "expert", "고급"]):
        return "숙련자"
    if any(k in s for k in ["경력", "experienced", "middle", "regular"]):
        return "경력자"
    if raw in ROLE_BASE_CAP:
        return raw
    return "경력자"


def map_driver_id_to_courier_id(driver_id: int) -> str:
    """
    ShimBox driverId → CSV courier_id 매핑
    - config.DRIVER_ID_TO_COURIER_ID에 있으면 그걸 사용
    - 없으면 "C_01", "C_02" 패턴으로 자동 생성
    """
    if driver_id in DRIVER_ID_TO_COURIER_ID:
        return DRIVER_ID_TO_COURIER_ID[driver_id]
    return f"C_{driver_id:02d}"
