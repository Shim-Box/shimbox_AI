import pandas as pd
import numpy as np
import os
import re
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

from .model import CONTEXT, CHANNELS



def normalize_phone(phone: str) -> str:
    if not phone:
        return ""
    return re.sub(r"[^\d]", "", str(phone))


def normalize_postal(postal: str) -> str:
    if not postal:
        return ""
    return re.sub(r"\s+", "", str(postal).strip())


def normalize_career(career: Optional[str]) -> str:
    if career in ["초보자", "경력자", "숙련자"]:
        return career
    return "기타"



def load_orders_from_excel(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, dtype=str)  # 숫자 앞자리 보존

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    required_cols = ["name", "phone", "postal", "address", "product_name", "qty"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"엑셀 파일에 필수 컬럼 누락: {missing}")

    df["phone"] = df["phone"].apply(normalize_phone)
    df["postal"] = df["postal"].apply(normalize_postal)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(1).astype(int)

    return df[required_cols]



def decide_target_count(career: Optional[str], health: Dict[str, Any]) -> int:
    base_target = 10

    if career == "초보자":
        base_target = 8
    elif career == "숙련자":
        base_target = 12

    finish3_response = health.get("finish3", 0)

    if finish3_response == -1:
        base_target = max(5, int(base_target * 0.7))
    elif finish3_response == 1:
        base_target = int(base_target * 1.1)

    return int(base_target)


def prepare_prediction_data(courier_id: str, date_target: str) -> Optional[np.ndarray]:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "raw"
    metrics_path = data_dir / "daily_metrics.csv"
    surveys_path = data_dir / "daily_surveys.csv"
    couriers_path = data_dir / "couriers.csv"

    if not (metrics_path.exists() and surveys_path.exists() and couriers_path.exists()):
        print("⚠️ 데이터 파일이 누락되어 있습니다. (metrics/surveys/couriers)")
        return None

    df_metrics = pd.read_csv(metrics_path)
    df_surveys = pd.read_csv(surveys_path)
    df_merged = pd.merge(df_metrics, df_surveys, on=["date", "courier_id"], how="inner")

    df_courier = pd.read_csv(couriers_path)
    role_map = {"초보자": 0, "경력자": 1, "숙련자": 2}
    df_courier["role_idx"] = df_courier["skill"].map(role_map).fillna(1).astype(int)

    df_target = df_merged[df_merged["courier_id"] == courier_id].copy()
    if df_target.empty:
        print(f"⚠️ courier_id {courier_id} 데이터 없음.")
        return None

    df_target["date"] = pd.to_datetime(df_target["date"])
    df_target = df_target.sort_values("date").reset_index(drop=True)

    target_dt = datetime.strptime(date_target, "%Y-%m-%d")
    start_dt = target_dt - timedelta(days=CONTEXT - 1)
    df_window = df_target[(df_target["date"] >= start_dt) & (df_target["date"] <= target_dt)]

    if len(df_window) < CONTEXT:
        print(f"⚠️ {courier_id}의 최근 {CONTEXT}일 데이터 부족.")
        return None

    role_idx = df_courier.loc[
        df_courier["courier_id"] == courier_id, "role_idx"
    ].squeeze() if courier_id in df_courier["courier_id"].values else 1

    df_window["role_idx"] = role_idx
    df_window["difficulty_dummy"] = 1.0

    # 파생 변수 계산
    df_window["time_per_delivery"] = np.where(
        df_window["deliveries"] > 0,
        (df_window["work_hours"] * 60) / df_window["deliveries"],
        np.nan,
    )
    df_window["deliveries_per_hour"] = np.where(
        df_window["work_hours"] > 0,
        df_window["deliveries"] / df_window["work_hours"],
        np.nan,
    )
    df_window["steps_per_hour"] = np.where(
        df_window["work_hours"] > 0,
        df_window["steps"] / df_window["work_hours"],
        np.nan,
    )
    df_window["hr_per_hour"] = np.where(
        df_window["work_hours"] > 0,
        df_window["avg_hr"] / df_window["work_hours"],
        np.nan,
    )

    df_window = df_window.fillna(df_window.mean(numeric_only=True)).fillna(0)

    X_sequence = df_window.tail(CONTEXT)[CHANNELS].to_numpy(dtype="float32")

    return X_sequence.reshape(1, CONTEXT, len(CHANNELS))
