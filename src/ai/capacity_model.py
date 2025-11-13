# src/ai/capacity_model.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import joblib
import pandas as pd


# 직군별 기본 물량
ROLE_BASE_CAP = {
    "초보자": 10,
    "경력자": 20,
    "숙련자": 30,
}


def normalize_career(v: Optional[str]) -> str:
    if not v:
        return "경력자"
    s = str(v).strip().lower()
    if any(k in s for k in ["초보", "신입", "beginner", "junior"]):
        return "초보자"
    if any(k in s for k in ["숙련", "senior", "expert", "고급"]):
        return "숙련자"
    if any(k in s for k in ["경력", "experienced", "middle", "regular"]):
        return "경력자"
    if s in ("초보자", "경력자", "숙련자"):
        return s
    return "경력자"


@dataclass
class CapacityModel:
    """
    - model: 학습된 회귀 모델 (옵션)
    - hist_df: courier별 과거 metrics+survey 머지된 데이터 (옵션)
    실제 추천은 career + finish3 규칙이 메인이고,
    model은 그냥 참고용(나중에 weight 조금 주고 싶으면 여기에만 수정하면 됨).
    """
    model: Optional[object] = None
    hist_df: Optional[pd.DataFrame] = None

    @classmethod
    def load(cls, model_dir: Optional[Path] = None) -> "CapacityModel":
        """
        models/capacity_model.pkl 이 있으면 로딩하고,
        없으면 None으로 둔다(룰베이스만 사용).
        """
        if model_dir is None:
            # project root 기준
            base_dir = Path(__file__).resolve().parents[2]
            model_dir = base_dir / "models"

        model_path = model_dir / "capacity_model.pkl"

        model = None
        hist_df = None

        if model_path.exists():
            try:
                model = joblib.load(model_path)
                print(f"[INFO] capacity_model 로딩 완료: {model_path}")
            except Exception as e:
                print(f"[WARN] capacity_model 로딩 실패: {e}")

        # 학습에 사용했던 merged DF는 꼭 필요하지 않아서 옵션으로 둠
        base_dir = Path(__file__).resolve().parents[2]
        merged_path = base_dir / "data" / "merged_for_capacity.csv"
        if merged_path.exists():
            try:
                hist_df = pd.read_csv(merged_path)
                print(f"[INFO] capacity_model용 merged DF 로딩 완료")
            except Exception as e:
                print(f"[WARN] merged DF 로딩 실패: {e}")

        return cls(model=model, hist_df=hist_df)

    # --- 내부: ML의 raw 예측(절대값) 얻기(지금은 로그용) ---
    def _predict_raw(self, courier_id: Optional[str]) -> Optional[float]:
        """
        과거 데이터 + ML 모델로 '내일 처리 건수' 같은 값을 예측.
        지금은 결과를 그대로 쓰지 않고, 그냥 참고용/로그용.
        """
        if self.model is None or self.hist_df is None or courier_id is None:
            return None

        try:
            df_c = self.hist_df[self.hist_df["courier_id"] == courier_id]
            if df_c.empty:
                return None

            # 아주 단순하게 마지막 하루 특징만 뽑아서 예측 (예시)
            last = df_c.sort_values("date").iloc[-1]

            feature_cols = [
                "work_hours",
                "deliveries",
                "bmi",
                "avg_hr",
                "steps",
                "load_rel",
                "strain",
                "wish",
            ]
            X = last[feature_cols].to_frame().T
            y_hat = float(self.model.predict(X)[0])
            return y_hat
        except Exception as e:
            print(f"[WARN] _predict_raw 실패: {e}")
            return None

    # --- 여기부터 진짜 중요한 함수: 최종 추천 물량 ---
    def recommend_capacity(
        self,
        *,
        career: Optional[str],
        health: Optional[Dict[str, Any]] = None,
        courier_id: Optional[str] = None,
    ) -> int:
        """
        최종 추천 물량 계산.

        1) career 기준 기본값: 초보자10 / 경력자20 / 숙련자30
        2) health.finish3(=퇴근 설문 3번)으로 강하게 조절:
           - "더 적게"  계열 → 0.6배 (초보자 10 → 6건)
           - "더 많이" 계열 → 1.6배 (초보자 10 → 16건)
           - 그 외 / None → 1.0배 (그대로)
        3) ML 모델이 있으면 raw_pred를 로그로만 찍어두고,
           원하면 여기에서 factor 조금 조정 가능.
        """

        base = ROLE_BASE_CAP[normalize_career(career)]
        finish3 = None
        if health:
            finish3 = str(health.get("finish3") or "").strip()

        # ① finish3가 없으면: 그냥 직군 기본값만
        if not finish3:
            print(f"[REC] health 없음 또는 finish3 없음 → career={normalize_career(career)} → base={base}")
            return base

        # ② finish3 해석
        s = finish3.replace(" ", "")
        if any(k in s for k in ["적게", "덜", "조금줄이"]):
            factor = 0.6   # 예: 10 → 6
        elif any(k in s for k in ["많이", "더많이", "좀더", "늘려"]):
            factor = 1.6   # 예: 10 → 16
        else:
            factor = 1.0   # "평소대로", "비슷했다" 등

        # ③ (옵션) ML 모델 참고 (지금은 로그만 남기고 안 씀)
        raw_pred = self._predict_raw(courier_id)
        if raw_pred is not None:
            print(f"[ML] courier_id={courier_id} 모델 raw 예측={raw_pred:.1f} (로그용, 최종값에는 직접 사용 안 함)")

        # ④ 최종 추천 = base * factor
        recommended = int(round(base * factor))

        # 안전 클램프: 최소 4건, 최대는 base*2 (초보자: 4~20, 경력자: 8~40, 숙련자: 12~60)
        recommended = max(4, min(recommended, base * 2))

        print(
            f"[REC] career={normalize_career(career)} base={base} "
            f"finish3='{finish3}' factor={factor} → 최종 추천={recommended}"
        )
        return recommended


# ==========================
#  외부용 헬퍼 함수
# ==========================

def predict_capacity_for_driver(
    driver_id: int,
    career_raw: Optional[str],
    health: Optional[Dict[str, Any]] = None,
) -> int:
    """
    assign_engine 등에서 쓰는 헬퍼.

    - driver_id: 현재는 ML courier_id와 직접 매핑 안 해서 로그용으로만 쓸 수 있음.
    - career_raw: 기사 DB에서 온 원본 career 문자열
    - health: /admin/driver/{id}/health 응답 dict

    실제 로직:
      1) CapacityModel.load()
      2) career + health.finish3 로 추천 물량 계산
      3) courier_id는 아직 None (ML은 로그만)
    """
    model = CapacityModel.load()
    # ML 로그에 driver_id를 그대로 써도 되고, 매핑 없으면 None으로 둬도 됨
    # 여기서는 일단 None으로 두고, 나중에 필요하면 str(driver_id) 등으로 바꿔도 OK.
    capacity = model.recommend_capacity(
        career=career_raw,
        health=health,
        courier_id=None,
    )
    return capacity
