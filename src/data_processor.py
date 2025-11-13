import pandas as pd
import numpy as np
import re
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

CONTEXT = 14
CHANNELS = [
    "deliveries", "work_hours", "steps", "avg_hr", "strain", "wish",
    "time_per_delivery", "deliveries_per_hour", "steps_per_hour", "hr_per_hour",
    "role_idx", "difficulty_dummy"
]

def normalize_phone(phone: str) -> str:
    if not phone: return ""
    return re.sub(r"[^\d]", "", str(phone))

def normalize_postal(postal: str) -> str:
    if not postal: return ""
    return re.sub(r"\s+", "", str(postal).strip())

def normalize_career(career: Optional[str]) -> str:
    if career in ["초보자", "경력자", "숙련자"]:
        return career
    return "기타"

def load_orders_from_excel(file_path: str) -> pd.DataFrame:
    df = pd.read_excel(file_path, dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    required_cols = ["name", "phone", "postal", "address", "product_name", "qty"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"엑셀 파일에 필수 컬럼 누락: {missing}")
    df["phone"] = df["phone"].apply(normalize_phone)
    df["postal"] = df["postal"].apply(normalize_postal)
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(1).astype(int)
    return df[required_cols]

def decide_target_count(career: Optional[str], health: Dict[str, Any]) -> int:
    base_target = {"초보자": 8, "경력자": 10, "숙련자": 12}.get(career or "경력자", 10)
    finish3 = health.get("finish3", 0)
    if finish3 == 1:
        return int(round(base_target * 1.6))
    if finish3 == -1:
        return max(1, int(round(base_target * 0.6)))
    return int(base_target)

def prepare_prediction_data(
    courier_id: int,
    date_target: str,
    couriers_df: pd.DataFrame,
    daily_metrics_df: pd.DataFrame,
    daily_surveys_df: pd.DataFrame,
) -> Optional[np.ndarray]:

    if daily_metrics_df is None or daily_surveys_df is None or couriers_df is None:
        return None

    for df in (daily_metrics_df, daily_surveys_df):
        if not df.empty and "courier_id" in df.columns:
            df["courier_id"] = df["courier_id"].apply(lambda x: int(str(x).split("_")[-1]) if pd.notna(x) else x)

    m = daily_metrics_df.copy()
    s = daily_surveys_df.copy()
    for col in ["deliveries", "work_hours", "steps", "avg_hr"]:
        if col not in m.columns: m[col] = np.nan
    for col in ["strain", "wish"]:
        if col not in s.columns: s[col] = np.nan

    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    s["date"] = pd.to_datetime(s["date"], errors="coerce")

    merged = pd.merge(
        m[["date", "courier_id", "deliveries", "work_hours", "steps", "avg_hr"]],
        s[["date", "courier_id", "strain", "wish"]],
        on=["date", "courier_id"], how="left",
    )
    merged = merged[merged["courier_id"] == int(courier_id)].copy()
    if merged.empty: return None
    merged = merged.sort_values("date").reset_index(drop=True)

    target_dt = datetime.strptime(date_target, "%Y-%m-%d")
    start_dt = target_dt - timedelta(days=CONTEXT - 1)
    win = merged[(merged["date"] >= start_dt) & (merged["date"] <= target_dt)].copy()
    if len(win) < CONTEXT: return None

    role_map = {"초보자": 0, "경력자": 1, "숙련자": 2}
    skill = None
    if "skill" in couriers_df.columns:
        row = couriers_df[couriers_df["courier_id"] == int(courier_id)]
        if not row.empty:
            skill = row["skill"].iloc[0]
    role_idx = role_map.get(skill, 1)

    win["role_idx"] = int(role_idx)
    win["difficulty_dummy"] = 1.0

    win["time_per_delivery"] = np.where(win["deliveries"] > 0, (win["work_hours"] * 60) / win["deliveries"], np.nan)
    win["deliveries_per_hour"] = np.where(win["work_hours"] > 0, win["deliveries"] / win["work_hours"], np.nan)
    win["steps_per_hour"] = np.where(win["work_hours"] > 0, win["steps"] / win["work_hours"], np.nan)
    win["hr_per_hour"] = np.where(win["work_hours"] > 0, win["avg_hr"] / win["work_hours"], np.nan)

    win = win.fillna(win.mean(numeric_only=True)).fillna(0)

    for c in CHANNELS:
        if c not in win.columns:
            win[c] = 0.0

    X = win.tail(CONTEXT)[CHANNELS].to_numpy(dtype="float32")
    return X.reshape(1, CONTEXT, len(CHANNELS))
