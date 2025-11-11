import pandas as pd
import numpy as np
import os
from math import radians, sin, cos, sqrt, atan2
from sklearn.metrics import mean_absolute_error
from typing import Optional, List, Dict, Tuple

from . import api_client, model, data_processor
from .model import SCALER_MIN, SCALER_RANGE

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def apply_policy(pred_qty: np.ndarray,
                 wish: np.ndarray,
                 min_qty: float,
                 max_multiplier: float=1.30,
                 wish_weight: float=0.03) -> np.ndarray:
    
    rec = pred_qty + wish_weight * wish * 100
    rec = np.maximum(rec, min_qty)
    
    return rec

def assign_to_zones(df_recommend: pd.DataFrame,
                    couriers_df: pd.DataFrame,
                    zones_df: pd.DataFrame,
                    round_unit:int=1) -> pd.DataFrame:
    
    pairs = []
    for _, c in couriers_df.iterrows():
        for _, z in zones_df.iterrows():
            d = haversine_km(c["home_lat"], c["home_lng"], z["zone_lat"], z["zone_lng"])
            pairs.append({"courier_id": c["courier_id"], "zone_id": z["zone_id"], "distance_km": d})
    dist_df = pd.DataFrame(pairs)

    tmp = df_recommend[["courier_id","a_star","strain","wish"]].copy()
    dist_df = dist_df.merge(tmp, on="courier_id", how="left")

    assignments = []
    remaining_by_courier = tmp.set_index("courier_id")["a_star"].to_dict()
    remaining_by_zone = zones_df.set_index("zone_id")["demand_qty"].to_dict()

    dist_df["priority_score"] = dist_df["distance_km"] - dist_df["strain"] * 10 - dist_df["wish"] * 5
    dist_df = dist_df.sort_values(by="priority_score", ascending=True)

    for zid in zones_df["zone_id"]:
        cand = dist_df[dist_df["zone_id"]==zid].copy()
        
        need = remaining_by_zone.get(zid, 0)
        if need <= 0:
            continue

        for _, row in cand.iterrows():
            cid = row["courier_id"]
            if need <= 0:
                break
            cap = remaining_by_courier.get(cid, 0)
            if cap <= 0:
                continue

            give = min(cap, need)
            give = int(give) // round_unit * round_unit
            if give <= 0:
                continue

            assignments.append({"courier_id": cid, "zone_id": zid, "assigned_qty": int(give)})
            remaining_by_courier[cid] -= give
            need -= give

        remaining_by_zone[zid] = need
        
    mae = mean_absolute_error(
        zones_df["demand_qty"].values, 
        pd.DataFrame(assignments).groupby('zone_id')['assigned_qty'].sum().reindex(zones_df["zone_id"], fill_value=0).values
    )
    
    return pd.DataFrame(assignments), mae


def run_pipeline(
    daily_metrics: pd.DataFrame,
    daily_surveys: pd.DataFrame,
    zones: pd.DataFrame,
    today_date: str,
    use_true_target: bool = False,
    login_info: Dict[str, str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[float]]:
    
    access_token = api_client.login_and_get_token(login_info)
    if not access_token:
        raise ValueError("외부 API 로그인 실패 (토큰 획득 실패)")

    couriers_df = api_client.get_approved_drivers(
        access_token, 
        allowed_attendance=['출근'], 
        allowed_conditions=['양호', '보통']
    )
    
    if couriers_df.empty:
        raise ValueError("배정 가능한 기사가 없습니다.")

    recommendations_list = []
    
    couriers_df['courier_id'] = couriers_df['courier_id'].apply(lambda x: int(x.split('_')[1]))
    daily_metrics['courier_id'] = daily_metrics['courier_id'].apply(lambda x: int(x.split('_')[1]))
    daily_surveys['courier_id'] = daily_surveys['courier_id'].apply(lambda x: int(x.split('_')[1]))
    
    for _, courier in couriers_df.iterrows():
        cid = courier['courier_id']
        
        X_sequence = data_processor.prepare_prediction_data(
            courier_id=cid, 
            date_target=today_date,
            couriers_df=couriers_df, 
            daily_metrics_df=daily_metrics,
            daily_surveys_df=daily_surveys
        )

        if X_sequence is None:
            pred_qty = data_processor.decide_target_count(courier.get('skill'), {})
            base_qty = daily_metrics[
                (daily_metrics['courier_id'] == cid) & 
                (daily_metrics['date'] == pd.to_datetime(today_date) - pd.Timedelta(days=1))
            ]['deliveries'].iloc[0] if not daily_metrics.empty else pred_qty
            
        else:
            pred_output_norm = model.patchtst_predict(X_sequence) 
            
            pred_qty = (pred_output_norm * SCALER_RANGE) + SCALER_MIN
            pred_qty = int(np.round(pred_qty[0]))
            
            base_qty = daily_metrics[
                (daily_metrics['courier_id'] == cid) & 
                (pd.to_datetime(daily_metrics['date']).dt.date == (pd.to_datetime(today_date) - pd.Timedelta(days=1)).date())
            ]['deliveries'].iloc[0] if not daily_metrics.empty else pred_qty

        
        wish = courier.get('wish', 0)
        strain = courier.get('strain', 0)
        
        a_star = apply_policy(
            np.array([pred_qty]), 
            np.array([wish]),
            min_qty=data_processor.decide_target_count(courier.get('skill'), {}),
            max_multiplier=1.30, 
            wish_weight=0.03
        )[0]
        
        max_limit = int(base_qty * 1.30)
        a_star = min(a_star, max_limit)
        
        recommendations_list.append({
            "date": today_date,
            "courier_id": cid,
            "today_qty": base_qty,
            "pred_qty_raw": pred_qty,
            "strain": strain,
            "wish": wish,
            "a_star": int(a_star),
            "rec_ratio": round(a_star / base_qty if base_qty > 0 else 1.0, 3)
        })

    recommendations = pd.DataFrame(recommendations_list)
    
    assignments, mae = assign_to_zones(
        recommendations, 
        couriers_df[["courier_id","home_lat","home_lng"]].drop_duplicates(subset=['courier_id']), 
        zones, 
        round_unit=1
    )
    
    return recommendations, assignments, mae


if __name__ == '__main__':
    pass