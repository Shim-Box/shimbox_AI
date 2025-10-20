import pandas as pd
import numpy as np
import joblib 
import os

# 수학 및 ML 관련 라이브러리 (run_pipeline 함수 내부에서 사용)
from math import radians, sin, cos, sqrt, atan2
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from typing import Optional, List, Dict

# 💡 중요 수정 사항: 데이터 전처리 함수는 이제 이 파일에 직접 정의되어 있거나,
# main.py 실행 시 모든 로직을 통합하므로, 불필요한 import를 제거했습니다.

# --- 1. 유틸리티 함수 (거리 계산, Feature Engineering, Z-Score) ---

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

DERIVED_FORMULAS = [
    ("time_per_delivery", lambda df: df["work_hours"] / df["deliveries"].replace(0, np.nan)),
    ("deliveries_per_hour", lambda df: df["deliveries"] / df["work_hours"].replace(0, np.nan)),
    ("steps_per_hour",  lambda df: df["steps"] / df["work_hours"].replace(0, np.nan)),
    ("steps_per_delivery", lambda df: df["steps"] / df["deliveries"].replace(0, np.nan)),
    ("hr_per_step", lambda df: df["avg_hr"] / df["steps"].replace(0, np.nan)),
    ("hr_per_hour", lambda df: df["avg_hr"] / df["work_hours"].replace(0, np.nan)),
]

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    for name, fn in DERIVED_FORMULAS:
        g[name] = fn(g)
    return g

def zscore_by_group(df: pd.DataFrame, group_col: str, cols: List[str], min_count:int=7) -> pd.DataFrame:
    df = df.copy()
    zcols = []
    for c in cols:
        zname = f"z_{c}"
        zcols.append(zname)
        df[zname] = np.nan

    for gid, sub in df.groupby(group_col):
        if len(sub) >= min_count:
            means = sub[cols].mean()
            stds = sub[cols].std(ddof=0).replace(0, 1.0)
            df.loc[sub.index, [f"z_{k}" for k in cols]] = (sub[cols] - means) / stds

    missing_mask = df[[f"z_{k}" for k in cols]].isna().any(axis=1)
    if missing_mask.any():
        gsub = df.loc[missing_mask, cols]
        gmeans = gsub.mean()
        gstds = gsub.std(ddof=0).replace(0,1.0)
        df.loc[missing_mask, [f"z_{k}" for k in cols]] = (gsub - gmeans) / gstds

    df[[f"z_{k}" for k in cols]] = df[[f"z_{k}" for k in cols]].fillna(0.0)
    return df

# --- 2. 모델 학습 및 예측 함수 ---

def train_ratio_model(df: pd.DataFrame,
                      features: List[str],
                      group_col: str="courier_id",
                      target_col: str="ratio") -> Dict:

    # NaN을 0.0으로 채워 모델 학습에 사용할 수 있도록 합니다. (이미 zscore에서 처리하지만 안전을 위해)
    X = df[features].fillna(0.0).astype(float) 
    y = df[target_col].astype(float)
    groups = df[group_col]

    n_splits = min(5, max(2, len(df[group_col].unique())))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(df))
    models = []
    
    # 모델 저장 폴더가 없으면 생성
    os.makedirs('models', exist_ok=True)
    MODEL_SAVE_PATH = 'models/capacity_ratio_predictor.pkl'

    print("   -> 모델 학습 시작...")
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        m = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42 + fold, # 폴드별 시드 변경
            n_jobs=-1
        )
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict(X.iloc[va])
        models.append(m)
        joblib.dump(m, f'{MODEL_SAVE_PATH.replace(".pkl", f"_{fold}.pkl")}')

    mae = mean_absolute_error(y, oof)
    print(f"   -> 모델 학습 완료. MAE (교차 검증 평균 절대 오차): {mae:.4f}")
    
    # 모든 모델을 한 리스트로 저장할 수도 있습니다. (선택 사항)
    # joblib.dump(models, MODEL_SAVE_PATH) 
    
    return {"models": models, "mae": mae, "features": features}


def predict_ratio(models, df: pd.DataFrame, features):
    X = df[features].fillna(0.0).astype(float) # 예측 시에도 NaN 처리
    preds = np.mean([m.predict(X) for m in models], axis=0)
    return preds


def apply_policy(pred_ratio: np.ndarray,
                 wish: np.ndarray,
                 min_ratio: float=0.80,
                 max_ratio: float=1.20,
                 wish_weight: float=0.03) -> np.ndarray:
    """
    예측된 비율(pred_ratio)에 정책(안전 범위, 희망값 반영)을 적용합니다.
    """
    # 1. 희망값 반영
    rec = pred_ratio + wish_weight * wish
    
    # 2. 안전 범위 적용 (80% ~ 120%)
    return np.clip(rec, min_ratio, max_ratio)


# --- 3. 지역 할당 함수 (운영 최적화) ---

def assign_to_zones(df_recommend: pd.DataFrame,
                    couriers_df: pd.DataFrame,
                    zones_df: pd.DataFrame,
                    round_unit:int=1) -> pd.DataFrame:
    """
    추천 물량(a_star)을 지역 수요와 기사 거리를 고려하여 할당합니다.
    """
    pairs = []
    for _, c in couriers_df.iterrows():
        for _, z in zones_df.iterrows():
            # 기사 집 위치와 지역 중심점 간 거리 계산
            d = haversine_km(c["home_lat"], c["home_lng"], z["zone_lat"], z["zone_lng"])
            pairs.append({"courier_id": c["courier_id"], "zone_id": z["zone_id"], "distance_km": d})
    dist_df = pd.DataFrame(pairs)

    tmp = df_recommend[["courier_id","a_star","strain","wish"]].copy()
    dist_df = dist_df.merge(tmp, on="courier_id", how="left")

    assignments = []
    remaining_by_courier = tmp.set_index("courier_id")["a_star"].to_dict()
    remaining_by_zone = zones_df.set_index("zone_id")["demand_qty"].to_dict()

    # 지역 수요를 채우기 위해, 거리가 가깝고(True), Strain이 낮으며(True), Wish가 높은(False) 기사 순으로 할당
    dist_df["priority_score"] = dist_df["distance_km"] - dist_df["strain"] * 10 - dist_df["wish"] * 5
    dist_df = dist_df.sort_values(by="priority_score", ascending=True)

    for zid in zones_df["zone_id"]:
        cand = dist_df[dist_df["zone_id"]==zid].copy()
        
        need = remaining_by_zone[zid]
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
        
    # 물량 총합 보정 (Water-fill 방식) - 전체 수요에 맞춰 조정 
    # **참고: 현재 로직은 지역별/기사별 최대치 내에서만 할당하며, 총 수요(total_demand)를 
    # 완벽하게 채우지 않을 수 있습니다. 이는 현실적인 제약조건을 따른 결과입니다.**

    return pd.DataFrame(assignments)


# --- 4. 메인 파이프라인 통합 함수 ---

def run_pipeline(
    daily_metrics: pd.DataFrame,
    daily_surveys: pd.DataFrame,
    couriers: pd.DataFrame,
    zones: pd.DataFrame,
    today_date: str,
    use_true_target: bool=False
):
    
    # 1. 데이터 병합 및 Feature Engineering
    df = daily_metrics.merge(daily_surveys, on=["date","courier_id"], how="left")
    df = add_derived_features(df)

    # 2. 개인화 Z-Score 계산
    numeric_cols = [
        "work_hours","deliveries","bmi","avg_hr","steps",
        "time_per_delivery","deliveries_per_hour","steps_per_hour","steps_per_delivery",
        "hr_per_step","hr_per_hour"
    ]
    df = zscore_by_group(df, "courier_id", numeric_cols, min_count=7)

    # 3. 학습/예측 데이터 분리
    d_today = pd.to_datetime(today_date).date()
    df_hist = df[pd.to_datetime(df["date"]).dt.date < d_today].copy() # 과거 데이터 (학습용)
    df_day = df[pd.to_datetime(df["date"]).dt.date == d_today].copy() # 오늘 데이터 (예측용)

    # 4. Target 변수 ('ratio') 정의
    if use_true_target and "next_day_qty" in df_hist.columns:
        df_hist["ratio"] = (df_hist["next_day_qty"] / df_hist["deliveries"].clip(lower=1)).clip(0.80,1.20)
    else:
        # 가상 데이터 환경에서는 휴리스틱 기반 비율을 Target으로 사용
        df_hist["ratio"] = (
            1.00 + 0.10*df_hist["load_rel"].fillna(0) # 긍정적 하중(load_rel)은 증가
                 - 0.15*df_hist["strain"].fillna(0)  # 무리(strain)는 감소
                 + 0.08*df_hist["wish"].fillna(0)    # 희망(wish)은 증가
        ).clip(0.80, 1.20) # 80%~120% 범위로 제한

    # 5. 모델 학습 (과거 데이터 사용)
    features = [f"z_{c}" for c in numeric_cols] + ["load_rel","strain","wish"]
    df_hist[features] = df_hist[features].fillna(0.0) # 학습 데이터의 NaN 처리

    print("\n--- 🧠 AI 모델 학습 시작 (Target: 처리 용량 비율) ---")
    model_info = train_ratio_model(df_hist, features, group_col="courier_id", target_col="ratio")
    
    # 6. 예측 (오늘 데이터 사용)
    df_day[features] = df_day[features].fillna(0.0) # 예측 데이터의 NaN 처리
    pred_ratio = predict_ratio(model_info["models"], df_day, features)

    # 7. 정책 적용 및 최종 추천 물량 산출
    wish_arr = df_day["wish"].fillna(0).values
    rec_ratio = apply_policy(pred_ratio, wish_arr, min_ratio=0.80, max_ratio=1.20, wish_weight=0.03)

    today_qty = df_day["deliveries"].fillna(df_day.get("today_qty", df_day["deliveries"])).values
    # 최종 추천 물량 (a_star) = 어제 처리 물량 * 정책 적용 비율
    a_star = np.round(today_qty * rec_ratio).astype(int)

    recommend = df_day[["date","courier_id","deliveries","load_rel","strain","wish"]].copy()
    recommend.rename(columns={"deliveries":"today_qty"}, inplace=True)
    recommend["pred_ratio"] = pred_ratio
    recommend["rec_ratio"] = rec_ratio
    recommend["a_star"] = a_star

    # 8. 지역 할당
    print("\n--- 🧭 지역별 수요 및 거리 기반 최종 할당 시작 ---")
    asg = assign_to_zones(recommend, couriers[["courier_id","home_lat","home_lng"]], zones, round_unit=1)
    
    return recommend, asg, model_info.get("mae", None)


if __name__ == '__main__':
    print("--- 🚚 기사 건강 맞춤 물류 배정 시스템 시작 (통합 실행) ---")
    
    # 1. 4가지 데이터 파일 로드
    try:
        couriers_df = pd.read_csv('data/raw/couriers.csv')
        zones_df = pd.read_csv('data/raw/zones.csv')
        daily_metrics_df = pd.read_csv('data/raw/daily_metrics.csv')
        daily_surveys_df = pd.read_csv('data/raw/daily_surveys.csv')
    except FileNotFoundError as e:
        print(f"❌ 오류: 필요한 데이터 파일이 없습니다. data_processor.py를 먼저 실행하세요. 오류 파일: {e}")
        exit()
        
    # 2. 파이프라인 실행
    # 학습 데이터의 마지막 날짜를 '오늘'로 가정하고 배정합니다.
    TODAY_DATE = daily_metrics_df['date'].max() 
    TOTAL_DEMAND = zones_df['demand_qty'].sum()
    
    recommendations, assignments, mae = run_pipeline(
        daily_metrics=daily_metrics_df,
        daily_surveys=daily_surveys_df,
        couriers=couriers_df,
        zones=zones_df,
        today_date=TODAY_DATE,
        use_true_target=False
    )

    print("\n--- ✅ 최종 배정 결과 요약 (내일) ---")
    print(f"총 물류 수요 (Zones 합계): {TOTAL_DEMAND}건")
    print(f"최종 할당된 물량 총합: {assignments['assigned_qty'].sum()}건")
    print(f"MA E (모델 평균 오차): {mae:.4f}")
    print("-" * 70)
    print("1. 개인별 추천 물량 (정책 반영된 적정 처리 용량):")
    # strain이 높을수록(1.0) a_star가 today_qty보다 낮게 나오는 것을 확인
    print(recommendations[['courier_id', 'today_qty', 'strain', 'wish', 'a_star', 'rec_ratio']].sort_values('a_star', ascending=False).head(10).to_string())
    print("\n2. 지역별 최종 할당 결과 (거리/수요 고려):")
    # 배정된 물량이 지역별 수요(zones_df)와 기사별 추천량(a_star)을 초과하지 않는지 확인
    print(assignments.groupby('zone_id')['assigned_qty'].sum().reset_index().rename(columns={'assigned_qty':'total_assigned_to_zone'}).merge(zones_df[['zone_id', 'demand_qty']], on='zone_id').to_string())
    print("-" * 70)
    print("시스템 종료. 최종 배정 결과를 확인하세요.")