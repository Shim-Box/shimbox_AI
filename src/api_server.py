import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from . import models, schemas, database, main, api_client
from datetime import date
from typing import Optional

app = FastAPI(
    title="📦 기사 건강 맞춤 물류 배정 시스템 API",
    description="외부 API(기사 데이터)와 내부 DB(활동 기록, 지역 정보)를 통합한 배정 서비스입니다. Swagger UI(/docs)를 통해 테스트할 수 있습니다.",
)

# DB 연결 및 테이블 생성
models.Base.metadata.create_all(bind=database.engine)


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/health", summary="API 상태 확인", status_code=status.HTTP_200_OK)
def health_check():
    """API 서버의 상태를 확인합니다."""
    return {"status": "ok", "message": "Courier Assignment System is running."}


@app.post("/run-assignment", response_model=schemas.PipelineResult, summary="AI 기반 물류 배정 실행")
def run_assignment(
    request: schemas.RunPipelineRequest,
    db: Session = Depends(get_db)
):
    """
    특정 날짜의 지역별 물류 수요를 기반으로 기사별 물량 추천 및 지역 할당을 실행합니다.
    """
    today_date_str = request.today_date.strftime('%Y-%m-%d')
    print(f"\n--- AI Pipeline 실행 요청 ({today_date_str}) ---")
    
    
    login_info = {"username": "admin", "password": "password"} 
    
    access_token = api_client.login_and_get_token(login_info)
    if not access_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="외부 API 로그인 실패 (토큰 획득 실패)")

    couriers_df = api_client.get_approved_drivers(
        access_token, 
        allowed_attendance=['출근'], 
        allowed_conditions=['양호', '보통']
    )
    
    if couriers_df.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="배정 가능한 기사가 없습니다. (필터 조건 또는 API 응답 확인)")

    try:
        daily_metrics_df = pd.read_sql_query(text("SELECT * FROM daily_metrics"), db.bind)
        daily_surveys_df = pd.read_sql_query(text("SELECT * FROM daily_surveys"), db.bind)
        
        zones_data = db.query(models.Zone).all()
        zones_df = pd.DataFrame([vars(z) for z in zones_data if not z.zone_id.startswith('_')]) # SQLAlchemy 객체를 DataFrame으로 변환

        demand_map = {d.zone_id: d.demand_qty for d in request.zone_demands}
        zones_df['demand_qty'] = zones_df['zone_id'].map(demand_map).fillna(0).astype(int)
        
        TOTAL_DEMAND = zones_df['demand_qty'].sum()
        
        recommendations, assignments, mae = main.run_pipeline(
            daily_metrics=daily_metrics_df,
            daily_surveys=daily_surveys_df,
            couriers=couriers_df,
            zones=zones_df,
            today_date=today_date_str,
            use_true_target=False
        )
        
        TOTAL_ASSIGNED = assignments['assigned_qty'].sum()
        
        db_assignments = []
        for _, row in assignments.iterrows():
            db_assignment = models.AssignmentResult(
                date=request.today_date,
                courier_id=int(row['courier_id']),
                zone_id=int(row['zone_id']),
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
        
    except Exception as e:
        db.rollback()
        print(f"❌ 오류 발생: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI 파이프라인 실행 중 치명적인 오류 발생: {str(e)}"
        )

