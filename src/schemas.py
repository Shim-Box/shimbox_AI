from pydantic import BaseModel
from typing import List, Optional
from datetime import date

#API 응답 스키마: 기사별 추천 물량 결과
class Recommendation(BaseModel):
    courier_id: int
    today_qty: int
    strain: float  # 계산된 최종 strain
    wish: float
    a_star: int    # 추천 물량
    rec_ratio: float

    class Config:
        orm_mode = True

class Assignment(BaseModel):
    courier_id: int
    zone_id: int
    assigned_qty: int

    class Config:
        orm_mode = True

class PipelineResult(BaseModel):
    mae: Optional[float]
    recommendations: List[Recommendation]
    assignments: List[Assignment]
    total_assigned_qty: int
    total_demand_qty: int

class ZoneDemand(BaseModel):
    zone_id: int
    demand_qty: int

class RunPipelineRequest(BaseModel):
    today_date: date
    zone_demands: List[ZoneDemand]