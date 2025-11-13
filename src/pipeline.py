import logging
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from . import data_processor
from .ml_model import SCALER_MIN, SCALER_RANGE, patchtst_predict
from .shimbox_client import get_approved_drivers  # 팀 Swagger 기반

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def apply_policy(pred_qty: np.ndarray, wish: np.ndarray, min_qty: int, max_multiplier: float = 1.30, wish_weight: float = 0.03) -> np.ndarray:
    rec = pred_qty + wish_weight * wish * 100
    rec = np.maximum(rec, min_qty)
    return rec

def assign_to_zones(df_recommend: pd.DataFrame, couriers_df: pd.DataFrame, zones_df: pd.DataFrame, round_unit: int = 1) -> Tuple[pd.DataFrame, float]:
    pairs = [
        {
            "courier_id": int(c["courier_id"]),
            "zone_id": int(z["zone_id"]),
            "distance_km": haversine_km(float(c["home_lat"]), float(c["home_lng"]), float(z["zone_lat"]), float(z["zone_lng"])),
        }
        for _, c in couriers_df.iterrows()
        for _, z in zones_df.iterrows()
    ]
    dist_df = pd.DataFrame(pairs)
    tmp = df_recommend[["courier_id", "a_star", "strain", "wish"]].copy()
    dist_df = dist_df.merge(tmp, on="courier_id", how="left")
    dist_df["priority_score"] = dist_df["distance_km"] - dist_df["strain"] * 10 - dist_df["wish"] * 5
    dist_df = dist_df.sort_values("priority_score")

    assignments: List[Dict] = []
    remaining_by_courier = tmp.set_index("courier_id")["a_star"].to_dict()
    remaining_by_zone = zones_df.set_index("zone_id")["demand_qty"].to_dict()

    for zid in zones_df["zone_id"]:
        need = int(remaining_by_zone.get(int(zid), 0))
        if need <= 0:
            continue
        for _, row in dist_df[dist_df["zone_id"] == zid].iterrows():
            cid = int(row["courier_id"])
            cap = int(remaining_by_courier.get(cid, 0))
            if need <= 0 or cap <= 0:
                continue
            give = min(cap, need)
            give = (int(give) // round_unit) * round_unit
            if give <= 0:
                continue
            assignments.append({"courier_id": cid, "zone_id": int(zid), "assigned_qty": give})
            remaining_by_courier[cid] = cap - give
            need -= give
        remaining_by_zone[int(zid)] = need

    assign_df = pd.DataFrame(assignments)
    if assign_df.empty:
        assigned_sum = pd.Series(0, index=zones_df["zone_id"])
    else:
        assigned_sum = assign_df.groupby("zone_id")["assigned_qty"].sum().reindex(zones_df["zone_id"], fill_value=0)
    mae = mean_absolute_error(zones_df["demand_qty"], assigned_sum)
    return assign_df, float(mae)

def run_pipeline(daily_metrics: pd.DataFrame, daily_surveys: pd.DataFrame, zones: pd.DataFrame, today_date: str, use_true_target: bool = False, login_info: Optional[Dict[str, str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    logging.info("승인 기사 목록 조회 중(ShimBox)…")
    # ShimBox 승인 기사 목록 (토큰은 shimbox_client쪽에서 Optional로 처리)
    couriers_df = get_approved_drivers(admin_token=None)
    if couriers_df.empty:
        raise ValueError("배정 가능한 기사가 없습니다.")

    logging.info(f"활성 기사 수: {len(couriers_df)}명")

    def safe_parse_id(x):
        try:
            return int(str(x).split("_")[-1])
        except Exception:
            return int(x) if str(x).isdigit() else np.nan

    # NOTE: 외부 승인 목록은 driverId 기준 → 내부 id로 변환해 사용
    couriers_df = couriers_df.rename(columns={"driverId": "courier_id"})
    couriers_df["courier_id"] = couriers_df["courier_id"].apply(safe_parse_id)

    for df in [daily_metrics, daily_surveys]:
        if not df.empty and "courier_id" in df.columns:
            df["courier_id"] = df["courier_id"].apply(safe_parse_id)

    recommendations: List[Dict] = []

    for _, courier in couriers_df.iterrows():
        cid = int(courier["courier_id"])

        X_seq = data_processor.prepare_prediction_data(
            courier_id=cid,
            date_target=today_date,
            couriers_df=couriers_df,
            daily_metrics_df=daily_metrics,
            daily_surveys_df=daily_surveys,
        )

        if X_seq is None:
            pred_qty = data_processor.decide_target_count(courier.get("career"), {})
            base_qty = int(daily_metrics.query("courier_id == @cid")["deliveries"].iloc[-1]) if (not daily_metrics.empty and (daily_metrics["courier_id"] == cid).any()) else int(pred_qty)
        else:
            pred_norm = patchtst_predict(X_seq)
            pred_qty = int(np.round(pred_norm * SCALER_RANGE + SCALER_MIN))
            base_qty = int(daily_metrics.query("courier_id == @cid")["deliveries"].iloc[-1]) if (not daily_metrics.empty and (daily_metrics["courier_id"] == cid).any()) else int(pred_qty)

        wish = float(0)
        strain = float(0)

        min_qty = int(data_processor.decide_target_count(courier.get("career"), {}))

        a_star_arr = apply_policy(np.array([pred_qty], dtype=float), np.array([wish], dtype=float), min_qty=min_qty, wish_weight=0.03)
        a_star = int(a_star_arr[0])
        if base_qty > 0:
            a_star = min(a_star, int(base_qty * 1.3))

        recommendations.append({
            "date": today_date,
            "courier_id": cid,
            "today_qty": int(base_qty),
            "pred_qty_raw": int(pred_qty),
            "strain": float(strain),
            "wish": float(wish),
            "a_star": int(a_star),
            "rec_ratio": round(a_star / base_qty, 3) if base_qty > 0 else 1.0,
        })

    rec_df = pd.DataFrame(recommendations)
    courier_geo = couriers_df[["courier_id"]].copy()
    # geo 좌표 없으면 거리 항목은 0으로 처리되므로, zones와 거리 우선순위는 의미가 약해질 수 있음
    courier_geo["home_lat"] = 37.5
    courier_geo["home_lng"] = 127.0

    assignments_df, mae = assign_to_zones(rec_df, courier_geo, zones)
    logging.info(f"✅ 파이프라인 완료 — MAE: {mae:.4f}")
    return rec_df, assignments_df, mae
