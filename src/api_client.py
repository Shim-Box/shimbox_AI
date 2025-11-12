# src/api_client.py
import os
import re
import requests
import pandas as pd
import numpy as np
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()  # ← .env 로드

API_BASE_URL = os.getenv("API_BASE_URL", "http://116.39.208.72:26443")
LOGIN_ENDPOINT = os.getenv("LOGIN_ENDPOINT", "")  # ← 기본값을 빈 문자열로
APPROVED_DRIVERS_ENDPOINT = os.getenv("APPROVED_DRIVERS_ENDPOINT", "/api/v1/admin/approved")
TIMEOUT = int(os.getenv("API_TIMEOUT", "8"))

def login_and_get_token(login_info: dict) -> Optional[str]:
    if not LOGIN_ENDPOINT:     # 로그인 엔드포인트 없으면 생략
        return None
    url = API_BASE_URL.rstrip("/") + LOGIN_ENDPOINT
    try:
        res = requests.post(url, json=login_info, timeout=TIMEOUT)
        res.raise_for_status()
        data = res.json()
        return data.get("access_token") or data.get("token") or data.get("jwt")
    except requests.RequestException:
        return None

def get_approved_drivers(
    access_token: Optional[str],
    allowed_attendance: Optional[List[str]] = None,
    allowed_conditions: Optional[List[str]] = None,
) -> pd.DataFrame:
    url = API_BASE_URL.rstrip("/") + APPROVED_DRIVERS_ENDPOINT
    headers = {"Authorization": f"Bearer {access_token}"} if access_token else {}
    params = {"page": 1, "size": 500}

    try:
        res = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
        res.raise_for_status()
        payload = res.json()
    except requests.RequestException:
        return pd.DataFrame(columns=[
            "courier_id","name","age","height","weight",
            "home_lat","home_lng","strain","wish","skill"
        ])

    items = payload.get("items") or payload.get("data") or payload.get("results") or payload
    df = pd.DataFrame(items)

    rename_map = {
        "id":"courier_id", "driverId":"courier_id", "courierId":"courier_id",
        "fullName":"name", "name":"name",
        "age":"age", "height":"height", "weight":"weight",
        "lat":"home_lat","lng":"home_lng","latitude":"home_lat","longitude":"home_lng",
        "homeLat":"home_lat","homeLng":"home_lng",
        "strain":"strain","fatigue":"strain",
        "wish":"wish","wishQty":"wish",
        "skill":"skill",
        "attendance":"attendance","condition":"condition",
        "approvalStatus":"approvalStatus","approval":"approvalStatus",
    }
    df = df.rename(columns=rename_map)

    # 필수 컬럼 보강
    required = ["courier_id","name","age","height","weight","home_lat","home_lng","strain","wish","skill"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    # 선택 필터
    if allowed_attendance and "attendance" in df.columns:
        df = df[df["attendance"].isin(allowed_attendance)]
    if allowed_conditions and "condition" in df.columns:
        df = df[df["condition"].isin(allowed_conditions)]
    if "approvalStatus" in df.columns:
        df = df[df["approvalStatus"] == True]

    # courier_id 정규화 → 정수
    def _safe_id(x):
        s = str(x)
        try:
            return int(s)
        except:
            m = re.findall(r"\d+", s)
            return int(m[0]) if m else np.nan
    df["courier_id"] = df["courier_id"].apply(_safe_id)

    # 숫자형 변환
    df["strain"] = pd.to_numeric(df["strain"], errors="coerce").clip(0, 10).fillna(0.0)  # ← 0~10
    df["wish"]   = pd.to_numeric(df["wish"], errors="coerce").fillna(100).astype(int)
    df["home_lat"] = pd.to_numeric(df["home_lat"], errors="coerce")
    df["home_lng"] = pd.to_numeric(df["home_lng"], errors="coerce")

    # 필수값 결측 제거 + 타입 확정
    df = df.dropna(subset=["courier_id"]).copy()
    df["courier_id"] = df["courier_id"].astype(int)

    return df.reindex(columns=required)
