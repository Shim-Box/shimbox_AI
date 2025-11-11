import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from . import models, schemas, database, main, api_client
from datetime import date
from typing import Optional

from . import model

app = FastAPI(
    title="📦 기사 건강 맞춤 물류 배정 시스템 API",
    description="외부 API(기사 데이터)와 내부 DB(활동 기록, 지역 정보)를 통합한 배정 서비스입니다. Swagger UI(/docs)를 통해 테스트할 수 있습니다.",
)

models.Base.metadata.create_all(bind=database.engine)

@app.on_event("startup")
def load_ai_model_on_startup():
    model.load_patchtst_model()
    print("[SERVER] PatchTST 모델이 API 서버 시작과 함께 미리 로드되었습니다.")


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health", summary="API 상태 확인", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "ok", "message": "Courier Assignment System is running."}


@app.post("/run-assignment", response_model=schemas.PipelineResult, summary="AI 기반 물류 배정 실행")
def run_assignment(
    request: schemas.RunPipelineRequest,
    db: Session = Depends(get_db)
):
    today_date_str = request.today_date.strftime('%Y-%m-%d')
    print(f"\n--- AI Pipeline 실행 요청 ({today_date_str}) ---")
    
    try:
        daily_metrics_df = pd.read_sql_query(text("SELECT * FROM daily_metrics"), db.bind)
        daily_surveys_df = pd.read_sql_query(text("SELECT * FROM daily_surveys"), db.bind)
        
        zones_data = db.query(models.Zone).all()
        zones_df = pd.DataFrame([
            {k: v for k, v in vars(z).items() if not k.startswith('_')} 
            for z in zones_data
        ])

        demand_map = {d.zone_id: d.demand_qty for d in request.zone_demands}
        zones_df['demand_qty'] = zones_df['zone_id'].map(demand_map).fillna(0).astype(int)
        
        TOTAL_DEMAND = zones_df['demand_qty'].sum()
        
        recommendations, assignments, mae = main.run_pipeline(
            daily_metrics=daily_metrics_df,
            daily_surveys=daily_surveys_df,
            zones=zones_df,
            today_date=today_date_str,
            use_true_target=False,
            login_info={"username": "admin", "password": "password"} 
        )
        
        TOTAL_ASSIGNED = assignments['assigned_qty'].sum()
        
        db_assignments = []
        for _, row in assignments.iterrows():
            db_assignment = models.AssignmentResult(
                date=request.today_date,
                courier_id=row['courier_id'], 
                zone_id=row['zone_id'],
                assigned_qty=int(row['assigned_qty'])
            )
            db_assignments.append(db_assignment)
            
        db.execute(text("DELETE FROM assignment_results WHERE date = :date"), {"date": request.today_date})
        db.bulk_save_objects(db_assignments)
        db.commit()
        
        print(f"--- 실행 완료. 총 수요: {TOTAL_DEMAND}, 총 할당: {TOTAL_ASSIGNED}, MAE: {mae} ---")
        
        return schemas.PipelineResult(
            mae=mae,
            recommendations=recommendations.to_dict('records'),
            assignments=assignments.to_dict('records'),
            total_assigned_qty=TOTAL_ASSIGNED,
            total_demand_qty=TOTAL_DEMAND
        )
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        import traceback
        traceback.print_exc()
        print(f"❌ 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI 파이프라인 실행 중 치명적인 오류 발생: {type(e).__name__}: {str(e)}"
        )