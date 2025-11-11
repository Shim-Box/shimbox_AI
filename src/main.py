import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics import mean_absolute_error
from typing import Optional, List, Dict, Tuple
import logging

from . import api_client, model, data_processor
from .model import SCALER_MIN, SCALER_RANGE

# 로깅 기본 설정
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def apply_policy(pred_qty, wish, min_qty, max_multiplier=1.30, wish_weight=0.03):
    """
    정책 기반 물량 조정
    """
    rec = pred_qty + wish_weight * wish * 100
    rec = np.maximum(rec, min_qty)
    return rec


def assign_to_zones(df_recommend, couriers_df, zones_df, round_unit=1):
    """
    거리 기반 zone 할당 + zone별 수요 충족률 계산
    """
    pairs = [
        {
            "courier_id": c["courier_id"],
            "zone_id": z["zone_id"],
            "distance_km": haversine_km(c["home_lat"], c["home_lng"], z["zone_lat"], z["zone_lng"])
        }
        for _, c in couriers_df.iterrows()
        for _, z in zones_df.iterrows()
    ]

    dist_df = pd.DataFrame(pairs)
    tmp = df_recommend[["courier_id", "a_star", "strain", "wish"]]
    dist_df = dist_df.merge(tmp, on="courier_id", how="left")

    dist_df["priority_score"] = dist_df["distance_km"] - dist_df["strain"] * 10 - dist_df["wish"] * 5
    dist_df = dist_df.sort_values("priority_score")

    assignments = []
    remaining_by_courier = tmp.set_index("courier_id")["a_star"].to_dict()
    remaining_by_zone = zones_df.set_index("zone_id")["demand_qty"].to_dict()

    for zid in zones_df["zone_id"]:
        need = remaining_by_zone.get(zid, 0)
        if need <= 0:
            continue

        for _, row in dist_df[dist_df["zone_id"] == zid].iterrows():
            cid = row["courier_id"]
            cap = remaining_by_courier.get(cid, 0)
            if need <= 0 or cap <= 0:
                continue

            give = min(cap, need)
            give = (int(give) // round_unit) * round_unit
            if give <= 0:
                continue

            assignments.append({"courier_id": cid, "zone_id": zid, "assigned_qty": give})
            remaining_by_courier[cid] -= give
            need -= give

        remaining_by_zone[zid] = need

    # zone별 MAE 계산
    assign_df = pd.DataFrame(assignments)
    assigned_sum = (
        assign_df.groupby("zone_id")["assigned_qty"].sum()
        .reindex(zones_df["zone_id"], fill_value=0)
        .fillna(0)
    )

    mae = mean_absolute_error(zones_df["demand_qty"], assigned_sum)
    return assign_df, mae


def run_pipeline(
    daily_metrics: pd.DataFrame,
    daily_surveys: pd.DataFrame,
    zones: pd.DataFrame,
    today_date: str,
    use_true_target: bool = False,
    login_info: Dict[str, str] = None,
):
    logging.info("로그인 토큰 요청 중...")
    access_token = api_client.login_and_get_token(login_info)
    if not access_token:
        raise ValueError("외부 API 로그인 실패 (토큰 획득 실패)")

    couriers_df = api_client.get_approved_drivers(
        access_token,
        allowed_attendance=["출근"],
        allowed_conditions=["양호", "보통"],
    )

    if couriers_df.empty:
        raise ValueError("배정 가능한 기사가 없습니다.")

    logging.info(f"활성 기사 수: {len(couriers_df)}명")

    def safe_parse_id(x):
        try:
            return int(str(x).split("_")[-1])
        except Exception:
            return int(x) if str(x).isdigit() else np.nan

    for df in [couriers_df, daily_metrics, daily_surveys]:
        df["courier_id"] = df["courier_id"].apply(safe_parse_id)

    recommendations = []

    for _, courier in couriers_df.iterrows():
        cid = courier["courier_id"]

        X_seq = data_processor.prepare_prediction_data(
            courier_id=cid,
            date_target=today_date,
            couriers_df=couriers_df,
            daily_metrics_df=daily_metrics,
            daily_surveys_df=daily_surveys,
        )

        if X_seq is None:
            pred_qty = data_processor.decide_target_count(courier.get("skill"), {})
            base_qty = (
                daily_metrics.query("courier_id == @cid")["deliveries"].iloc[-1]
                if not daily_metrics.empty else pred_qty
            )
        else:
            pred_norm = model.patchtst_predict(X_seq)
            pred_qty = int(np.round(pred_norm * SCALER_RANGE + SCALER_MIN))
            base_qty = (
                daily_metrics.query("courier_id == @cid")["deliveries"].iloc[-1]
                if not daily_metrics.empty else pred_qty
            )

        wish = courier.get("wish", 0)
        strain = courier.get("strain", 0)

        a_star = apply_policy(
            np.array([pred_qty]), np.array([wish]),
            min_qty=data_processor.decide_target_count(courier.get("skill"), {}),
            wish_weight=0.03
        )[0]

        a_star = min(a_star, int(base_qty * 1.3))

        recommendations.append({
            "date": today_date,
            "courier_id": cid,
            "today_qty": base_qty,
            "pred_qty_raw": pred_qty,
            "strain": strain,
            "wish": wish,
            "a_star": int(a_star),
            "rec_ratio": round(a_star / base_qty if base_qty > 0 else 1.0, 3),
        })

    rec_df = pd.DataFrame(recommendations)
    assignments, mae = assign_to_zones(
        rec_df,
        couriers_df[["courier_id", "home_lat", "home_lng"]].drop_duplicates(),
        zones,
    )

    logging.info(f"✅ 파이프라인 완료 — MAE: {mae:.4f}")
    return rec_df, assignments, mae
