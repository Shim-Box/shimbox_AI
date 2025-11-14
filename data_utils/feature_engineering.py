import numpy as np
import pandas as pd

SEQ_LEN = 7  # PatchTST에 넣을 최근 일수

# -----------------------------
# 공통 유틸
# -----------------------------
def add_bmi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2
    return df

def encode_finish(x: str) -> int:
    mapping = {
        "적었다": 0,
        "비슷했다": 1,
        "많았다": 2,
        "전혀 아니다": 0,
        "약간 그렇다": 1,
        "매우 그렇다": 2,
        "적게": 0,
        "평소대로": 1,
        "더 많이": 2,
    }
    return mapping.get(x, 1)

def encode_condition(x: str) -> int:
    mapping = {"위험": 0, "불안": 1, "좋음": 2}
    return mapping.get(x, 1)

def encode_career(x: str) -> int:
    mapping = {"beginner": 0, "experienced": 1, "expert": 2}
    return mapping.get(x, 1)

# -----------------------------
# PatchTST 학습용
# -----------------------------
def build_patchtst_dataset(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    df: 한 번에 모든 driver 데이터 (train_history.csv)
    return: X (num_samples, seq_len, 4), y (num_samples,)
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["driverId", "date"])

    features = []
    targets = []

    for driver_id, g in df.groupby("driverId"):
        g = g.sort_values("date")
        metrics = g[["steps", "avg_hr", "work_hours", "deliveries"]].values
        caps = g["capacity_label"].values

        if len(g) <= seq_len:
            continue

        for i in range(seq_len, len(g)):
            seq = metrics[i - seq_len : i]  # 마지막 seq_len일
            y = caps[i]  # 다음날 capacity
            features.append(seq)
            targets.append(y)

    X = np.stack(features)
    y = np.array(targets, dtype=np.float32)
    return X, y

# -----------------------------
# RandomForest 학습용
# -----------------------------
def build_rf_dataset(df: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    하루당 한 행의 feature + capacity_label
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = add_bmi(df)
    df = df.sort_values(["driverId", "date"])

    rows = []

    for driver_id, g in df.groupby("driverId"):
        g = g.sort_values("date")
        for i in range(len(g)):
            if i < seq_len:
                continue  # 최근 seq_len일 이전은 건너뛰기

            window = g.iloc[i - seq_len : i]
            today = g.iloc[i]

            feat = {
                "driverId": driver_id,
                "bmi": today["bmi"],
                "career_enc": encode_career(today["career"]),
                "finish1_enc": encode_finish(today["finish1"]),
                "finish2_enc": encode_finish(today["finish2"]),
                "finish3_enc": encode_finish(today["finish3"]),
                "condition_enc": encode_condition(today["conditionStatus"]),
                "steps_mean_7": window["steps"].mean(),
                "work_hours_mean_7": window["work_hours"].mean(),
                "deliveries_mean_7": window["deliveries"].mean(),
                "deliveries_std_7": window["deliveries"].std(),
                "target_capacity": today["capacity_label"],
            }
            rows.append(feat)

    feat_df = pd.DataFrame(rows)
    y = feat_df.pop("target_capacity").values.astype("float32")
    X = feat_df.drop(columns=["driverId"]).values.astype("float32")
    return X, y, feat_df.columns.tolist()
