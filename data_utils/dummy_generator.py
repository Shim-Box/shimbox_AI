import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils.logger import get_logger

logger = get_logger("dummy_generator")

CAREER_LEVELS = ["beginner", "experienced", "expert"]

def generate_dummy_data(
    num_drivers: int = 5,
    num_days: int = 90,
    start_date: str = "2025-01-01",
    out_path: str = "data/train_history.csv",
):
    start = datetime.fromisoformat(start_date)
    rows = []

    rng = np.random.default_rng(42)

    for driver_id in range(1, num_drivers + 1):
        career = CAREER_LEVELS[(driver_id - 1) % len(CAREER_LEVELS)]
        height = rng.integers(160, 185)
        weight = rng.integers(55, 90)

        base_deliveries = {
            "beginner": 10,
            "experienced": 20,
            "expert": 30,
        }[career]

        for d in range(num_days):
            date = start + timedelta(days=d)

            # 요일에 따라 약간 변동
            weekday = date.weekday()  # 0=월
            weekend_factor = 0.8 if weekday >= 5 else 1.0

            deliveries = int(
                rng.normal(base_deliveries * weekend_factor, base_deliveries * 0.2)
            )
            deliveries = max(4, deliveries)

            work_hours = rng.normal(4 + base_deliveries / 10, 0.5)
            work_hours = max(2, min(work_hours, 10))

            steps = int(rng.normal(8000 + deliveries * 200, 1500))
            steps = max(2000, steps)

            avg_hr = int(rng.normal(80 + deliveries * 0.5, 5))

            # 피로/컨디션
            if work_hours > 7 or deliveries > base_deliveries * 1.3:
                finish2 = "매우 그렇다"
                condition = "불안"
            elif work_hours > 5:
                finish2 = "약간 그렇다"
                condition = "좋음"
            else:
                finish2 = "전혀 아니다"
                condition = "좋음"

            # 오늘 물량이 평소보다?
            if deliveries < base_deliveries * 0.8:
                finish1 = "적었다"
            elif deliveries > base_deliveries * 1.2:
                finish1 = "많았다"
            else:
                finish1 = "비슷했다"

            # 내일 희망
            if finish2 == "매우 그렇다":
                finish3 = "적게"
            elif finish1 == "적었다" and finish2 == "전혀 아니다":
                finish3 = "더 많이"
            else:
                finish3 = "평소대로"

            # 우리가 예측하려는 "내일 적정 물량(capacity_label)"
            # 간단하게 오늘 deliveries에 약간의 노이즈 + 설문 반영
            capacity = deliveries
            if finish3 == "적게":
                capacity = max(4, int(deliveries * 0.7))
            elif finish3 == "더 많이":
                capacity = int(deliveries * 1.2)
            capacity = max(4, capacity)

            rows.append(
                {
                    "driverId": driver_id,
                    "date": date.date().isoformat(),
                    "career": career,
                    "height": height,
                    "weight": weight,
                    "steps": steps,
                    "avg_hr": avg_hr,
                    "work_hours": work_hours,
                    "deliveries": deliveries,
                    "finish1": finish1,
                    "finish2": finish2,
                    "finish3": finish3,
                    "conditionStatus": condition,
                    "capacity_label": capacity,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"✅ 더미 데이터 생성 완료: {out_path}, rows={len(df)}")

if __name__ == "__main__":
    generate_dummy_data()
