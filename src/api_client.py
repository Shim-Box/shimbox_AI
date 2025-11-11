import requests
import pandas as pd
import logging
from typing import Optional, List
import random
import numpy as np

try:
    from config.settings import API_BASE_URL, LOGIN_ENDPOINT, APPROVED_DRIVERS_ENDPOINT
except ImportError:
    API_BASE_URL = "http://api.external-service.com"
    LOGIN_ENDPOINT = "/auth/login"
    APPROVED_DRIVERS_ENDPOINT = "/api/v1/admin/approved"
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def login_and_get_token(login_info: dict) -> Optional[str]:
    """관리자 로그인 후 access token 반환"""
    if login_info.get("username") == "admin":
          return "fake_access_token_12345" 
    return None


def _create_dummy_couriers(num_couriers: int = 20) -> pd.DataFrame:
    """AI 로직 테스트를 위한 더미 기사 데이터 생성 함수 (실제 필드 이름 반영)."""
    courier_ids = [f"courier_{i}" for i in list(range(1001, 1001 + num_couriers))]
    data = {
        'driverId': courier_ids,
        'name': [f"기사{i}" for i in range(1, num_couriers + 1)],
        'age': [random.randint(25, 55) for _ in range(num_couriers)],
        'height': [random.uniform(160, 185) for _ in range(num_couriers)],
        'weight': [random.uniform(55, 90) for _ in range(num_couriers)], 
        'home_lat': [random.uniform(37.4, 37.6) for _ in range(num_couriers)],
        'home_lng': [random.uniform(126.9, 127.1) for _ in range(num_couriers)],
        'approvalStatus': [True] * num_couriers,
        'averageWorking': [random.uniform(7.0, 10.0) for _ in range(num_couriers)], 
        'averageDelivery': [random.randint(80, 150) for _ in range(num_couriers)],
        'attendance': random.choices(['출근', '휴가', '결근'], k=num_couriers, weights=[0.8, 0.15, 0.05]), 
        'conditionStatus': random.choices(['양호', '보통', '나쁨'], k=num_couriers, weights=[0.6, 0.3, 0.1]), 
        'strain': [random.uniform(0.1, 0.9) for _ in range(num_couriers)], # 추가: 피로도
        'wish': [random.randint(50, 150) for _ in range(num_couriers)], # 추가: 희망 물량
        'skill': random.choices(['초보자', '경력자', '숙련자'], k=num_couriers, weights=[0.2, 0.5, 0.3]), # 추가: 숙련도
    }
    return pd.DataFrame(data)


def get_approved_drivers(access_token: str, allowed_attendance: Optional[List[str]] = None, allowed_conditions: Optional[List[str]] = None) -> pd.DataFrame:
    """
    외부 API를 통해 기사 목록을 가져오고 필터링합니다.
    """
    url = API_BASE_URL + APPROVED_DRIVERS_ENDPOINT
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"page": 1, "size": 100}

    try:
        logging.warning("⚠️ API 호출 대신 테스트용 더미 기사 데이터를 사용합니다.")
        df = _create_dummy_couriers()
        
    except requests.RequestException as e:
        logging.error(f"❌ 기사 목록 조회 실패: {e}")
        return pd.DataFrame()
    

    rename_map = {
        "driverId": "courier_id",
        "conditionStatus": "condition",
        "averageWorking": "workTime",
        "averageDelivery": "deliveryStats",
    }
    df = df.rename(columns=rename_map)
    
    df = df[df["approvalStatus"] == True]

    if allowed_attendance:
        df = df[df["attendance"].isin(allowed_attendance) | df["attendance"].isna()]
    if allowed_conditions:
        df = df[df["condition"].isin(allowed_conditions) | df["condition"].isna()]

    logging.info(f"✅ 필터링 후 배정 가능 기사: {len(df)}명")
    
    # main.py에서 필요로 하는 최종 컬럼 목록을 맞추고, 외부 API에서 추가로 제공될 수 있는 strain, wish, skill도 포함
    required_cols = ['courier_id', 'name', 'age', 'height', 'weight', 'home_lat', 'home_lng', 'strain', 'wish', 'skill']
    
    # 누락된 컬럼에 대한 처리 및 최종 필터링
    final_df = df.reindex(columns=required_cols).copy()
    
    return final_df.dropna(subset=['courier_id'])