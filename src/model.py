from sqlalchemy import Column, Integer, String, Float, Date, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base  # src/database.py의 Base 클래스 임포트

#기사 정보 테이블
class Courier(Base):
    tablename = "couriers"

    courier_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    age = Column(Integer)
    height = Column(Float)
    weight = Column(Float)
    home_lat = Column(Float)
    home_lng = Column(Float)

#관계 설정 (Metric, Survey에 연결)
    metrics = relationship("DailyMetric", back_populates="courier")
    surveys = relationship("DailySurvey", back_populates="courier")
    assignments = relationship("AssignmentResult", back_populates="courier")


#일일 활동 테이블
class DailyMetric(Base):
    tablename = "daily_metrics"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    courier_id = Column(Integer, ForeignKey("couriers.courier_id"))

    work_hours = Column(Float)
    deliveries = Column(Integer)
    avg_hr = Column(Float)
    steps = Column(Integer)

    courier = relationship("Courier", back_populates="metrics")


#일일 설문 테이블 (피로도/희망 물량)
class DailySurvey(Base):
    tablename = "daily_surveys"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    courier_id = Column(Integer, ForeignKey("couriers.courier_id"))

    strain = Column(Float) 
    wish = Column(Float)   

    courier = relationship("Courier", back_populates="surveys")


#지역 정보 테이블
class Zone(Base):
    tablename = "zones"

    zone_id = Column(Integer, primary_key=True, index=True)
    zone_name = Column(String)
    zone_lat = Column(Float)
    zone_lng = Column(Float)

#당일 수요 (AI 실행 시 임시로 사용)
    demand_qty = Column(Integer)


#최종 배정 결과 테이블 (API 결과 저장)
class AssignmentResult(Base):
    tablename = "assignment_results"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    courier_id = Column(Integer, ForeignKey("couriers.courier_id"))
    zone_id = Column(Integer, ForeignKey("zones.zone_id"))
    assigned_qty = Column(Integer)

    courier = relationship("Courier", back_populates="assignments")
    zone = relationship("Zone")