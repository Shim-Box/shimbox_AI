import os
import logging
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text, select
from sqlalchemy.orm import Session

from .database import get_db
from .model import Zone, AssignmentResult
from . import schemas
from . import pipeline

# (선택) ShimBox 워크플로 API로 감싸서 실행
from .workflows.reallocate import main as run_reallocation

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

router = APIRouter(prefix="/api", tags=["📦 배정 파이프라인"])

@router.get("/health", summary="API 상태 확인", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "ok", "message": "Courier Assignment System is running."}

def _zones_df(db: Session) -> pd.DataFrame:
    rows = db.execute(select(Zone.zone_id, Zone.zone_lat, Zone.zone_lng, Zone.demand_qty)).all()
    return pd.DataFrame(rows, columns=["zone_id", "zone_lat", "zone_lng", "demand_qty"])

@router.post("/run-assignment", response_model=schemas.PipelineResult, summary="AI 기반 물류 배정 실행")
def run_assignment(request: schemas.RunPipelineRequest, db: Session = Depends(get_db)):
    today_date_str = request.today_date.strftime("%Y-%m-%d")
    logging.info(f"🚀 AI Pipeline 실행 요청: {today_date_str}")

    try:
        try:
            daily_metrics_df = pd.read_sql_query(text("SELECT * FROM daily_metrics"), db.bind)
            daily_surveys_df = pd.read_sql_query(text("SELECT * FROM daily_surveys"), db.bind)
        except Exception:
            logging.warning("⚠️ metrics 또는 surveys 테이블이 비어 있습니다.")
            daily_metrics_df, daily_surveys_df = pd.DataFrame(), pd.DataFrame()

        zones_df = _zones_df(db)
        if zones_df.empty:
            raise HTTPException(status_code=404, detail="Zone 데이터가 없습니다. 먼저 지역 데이터를 등록하세요.")

        demand_map = {d.zone_id: d.demand_qty for d in request.zone_demands}
        zones_df["demand_qty"] = zones_df["zone_id"].map(demand_map).fillna(0).astype(int)
        total_demand = int(zones_df["demand_qty"].sum())

        login_info = {
            "username": os.getenv("API_USERNAME", "admin"),
            "password": os.getenv("API_PASSWORD", "password"),
        }

        rec_df, assign_df, mae = pipeline.run_pipeline(
            daily_metrics=daily_metrics_df,
            daily_surveys=daily_surveys_df,
            zones=zones_df,
            today_date=today_date_str,
            use_true_target=False,
            login_info=login_info,
        )

        total_assigned = int(assign_df["assigned_qty"].sum()) if not assign_df.empty else 0

        db.execute(text("DELETE FROM assignment_results WHERE date = :date"), {"date": request.today_date})
        if not assign_df.empty:
            db.bulk_save_objects([
                AssignmentResult(
                    date=request.today_date,
                    courier_id=int(row["courier_id"]),
                    zone_id=int(row["zone_id"]),
                    assigned_qty=int(row["assigned_qty"]),
                )
                for _, row in assign_df.iterrows()
            ])
        db.commit()

        logging.info(f"✅ 파이프라인 완료: 수요={total_demand}, 할당={total_assigned}, MAE={mae:.4f}")

        return schemas.PipelineResult(
            mae=float(mae),
            recommendations=rec_df.to_dict("records"),
            assignments=assign_df.to_dict("records") if not assign_df.empty else [],
            total_assigned_qty=total_assigned,
            total_demand_qty=total_demand,
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logging.exception("❌ AI 파이프라인 실행 중 오류 발생")
        raise HTTPException(status_code=500, detail=f"AI 파이프라인 실행 실패: {type(e).__name__}: {str(e)}")

@router.post("/reallocate", summary="ShimBox 재배정 워크플로 실행")
def reallocate():
    run_reallocation()
    return {"status": "ok"}
