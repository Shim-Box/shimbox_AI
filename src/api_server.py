import pandas as pd
import logging
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import date
from typing import Optional
import os

from . import models, schemas, database, main, api_client, model

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

app = FastAPI(
    title="📦 기사 건강 맞춤 물류 배정 시스템 API",
    description="기사 건강데이터 기반 AI 배정 시스템.\nSwagger UI: `/docs`",
)

# --- DB 초기화 (마이그레이션 도구 사용 전까지만 허용) ---
@app.on_event("startup")
def on_startup():
    models.Base.metadata.create_all(bind=database.engine)
    model.load_patchtst_model()
    logging.info("✅ PatchTST 모델이 서버 시작 시 로드되었습니다.")


# --- DB 세션 종속성 ---
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Health Check ---
@app.get("/health", summary="API 상태 확인", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "ok", "message": "Courier Assignment System is running."}


# --- 핵심 파이프라인 실행 API ---
@app.post("/run-assignment", response_model=schemas.PipelineResult, summary="AI 기반 물류 배정 실행")
def run_assignment(request: schemas.RunPipelineRequest, db: Session = Depends(get_db)):
    today_date_str = request.today_date.strftime("%Y-%m-%d")
    logging.info(f"🚀 AI Pipeline 실행 요청: {today_date_str}")

    try:
        # --- DB에서 데이터 로드 ---
        try:
            daily_metrics_df = pd.read_sql_query(text("SELECT * FROM daily_metrics"), db.bind)
            daily_surveys_df = pd.read_sql_query(text("SELECT * FROM daily_surveys"), db.bind)
        except Exception:
            logging.warning("⚠️ metrics 또는 surveys 테이블이 비어있습니다.")
            daily_metrics_df, daily_surveys_df = pd.DataFrame(), pd.DataFrame()

        zones_data = db.query(models.Zone).all()
        if not zones_data:
            raise HTTPException(status_code=404, detail="Zone 데이터가 없습니다. 먼저 지역 데이터를 등록하세요.")

        zones_df = pd.DataFrame([{k: v for k, v in vars(z).items() if not k.startswith("_")} for z in zones_data])

        # --- 요청에서 수요량 반영 ---
        demand_map = {d.zone_id: d.demand_qty for d in request.zone_demands}
        zones_df["demand_qty"] = zones_df["zone_id"].map(demand_map).fillna(0).astype(int)
        TOTAL_DEMAND = int(zones_df["demand_qty"].sum())

        # --- AI 파이프라인 실행 ---
        login_info = {
            "username": os.getenv("API_USERNAME", "admin"),
            "password": os.getenv("API_PASSWORD", "password"),
        }

        recommendations, assignments, mae = main.run_pipeline(
            daily_metrics=daily_metrics_df,
            daily_surveys=daily_surveys_df,
            zones=zones_df,
            today_date=today_date_str,
            use_true_target=False,
            login_info=login_info,
        )

        TOTAL_ASSIGNED = int(assignments["assigned_qty"].sum())

        # --- 기존 동일 날짜 데이터 삭제 후 저장 ---
        db.execute(text("DELETE FROM assignment_results WHERE date = :date"), {"date": request.today_date})
        db.bulk_save_objects([
            models.AssignmentResult(
                date=request.today_date,
                courier_id=row["courier_id"],
                zone_id=row["zone_id"],
                assigned_qty=int(row["assigned_qty"]),
            )
            for _, row in assignments.iterrows()
        ])
        db.commit()

        logging.info(f"✅ 파이프라인 완료: 총 수요={TOTAL_DEMAND}, 할당={TOTAL_ASSIGNED}, MAE={mae:.4f}")

        return schemas.PipelineResult(
            mae=mae,
            recommendations=recommendations.to_dict("records"),
            assignments=assignments.to_dict("records"),
            total_assigned_qty=TOTAL_ASSIGNED,
            total_demand_qty=TOTAL_DEMAND,
        )

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logging.exception("❌ AI 파이프라인 실행 중 오류 발생")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI 파이프라인 실행 실패: {type(e).__name__}: {str(e)}",
        )
