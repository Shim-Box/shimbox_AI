from sqlalchemy import (
    Column, Integer, String, Float, Date, ForeignKey, UniqueConstraint, Index, CheckConstraint
)
from sqlalchemy.orm import relationship
from .database import Base

# 기사 정보 테이블
class Courier(Base):
    __tablename__ = "couriers"

    courier_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    age = Column(Integer, nullable=True)
    height = Column(Float, nullable=True)
    weight = Column(Float, nullable=True)
    home_lat = Column(Float, nullable=True)
    home_lng = Column(Float, nullable=True)

    # 관계 설정
    metrics = relationship(
        "DailyMetric",
        back_populates="courier",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    surveys = relationship(
        "DailySurvey",
        back_populates="courier",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    assignments = relationship(
        "AssignmentResult",
        back_populates="courier",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        CheckConstraint("age IS NULL OR age >= 16", name="ck_couriers_age_min"),
        CheckConstraint("height IS NULL OR height > 0", name="ck_couriers_height_pos"),
        CheckConstraint("weight IS NULL OR weight > 0", name="ck_couriers_weight_pos"),
    )


# 일일 활동 테이블
class DailyMetric(Base):
    __tablename__ = "daily_metrics"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True, nullable=False)
    courier_id = Column(
        Integer,
        ForeignKey("couriers.courier_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    work_hours = Column(Float, nullable=False, default=0.0)
    deliveries = Column(Integer, nullable=False, default=0)
    avg_hr = Column(Float, nullable=True)
    steps = Column(Integer, nullable=True, default=0)

    courier = relationship("Courier", back_populates="metrics")

    __table_args__ = (
        UniqueConstraint("date", "courier_id", name="uq_daily_metrics_date_courier"),
        CheckConstraint("work_hours >= 0 AND work_hours <= 24", name="ck_dm_work_hours"),
        CheckConstraint("deliveries >= 0", name="ck_dm_deliveries_nonneg"),
        CheckConstraint("steps IS NULL OR steps >= 0", name="ck_dm_steps_nonneg"),
        Index("ix_daily_metrics_courier_date", "courier_id", "date"),
    )


# 일일 설문 테이블 (피로도/희망 물량)
class DailySurvey(Base):
    __tablename__ = "daily_surveys"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True, nullable=False)
    courier_id = Column(
        Integer,
        ForeignKey("couriers.courier_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    strain = Column(Float, nullable=True) 
    wish = Column(Float, nullable=True)

    courier = relationship("Courier", back_populates="surveys")

    __table_args__ = (
        UniqueConstraint("date", "courier_id", name="uq_daily_surveys_date_courier"),
        CheckConstraint("strain IS NULL OR (strain >= 0 AND strain <= 10)", name="ck_ds_strain_range"),
        CheckConstraint("wish IS NULL OR wish >= 0", name="ck_ds_wish_nonneg"),
        Index("ix_daily_surveys_courier_date", "courier_id", "date"),
    )


# 지역 정보 테이블
class Zone(Base):
    __tablename__ = "zones"

    zone_id = Column(Integer, primary_key=True, index=True)
    zone_name = Column(String, nullable=False, unique=True)
    zone_lat = Column(Float, nullable=True)
    zone_lng = Column(Float, nullable=True)

    # 당일 수요
    demand_qty = Column(Integer, nullable=True, default=None)


# 최종 배정 결과 테이블 (API 결과 저장)
class AssignmentResult(Base):
    __tablename__ = "assignment_results"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True, nullable=False)
    courier_id = Column(
        Integer,
        ForeignKey("couriers.courier_id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    zone_id = Column(
        Integer,
        ForeignKey("zones.zone_id", ondelete="RESTRICT"),
        index=True,
        nullable=False,
    )
    assigned_qty = Column(Integer, nullable=False, default=0)

    courier = relationship("Courier", back_populates="assignments")
    zone = relationship("Zone")

    __table_args__ = (
        UniqueConstraint("date", "courier_id", "zone_id", name="uq_assignments_date_courier_zone"),
        CheckConstraint("assigned_qty >= 0", name="ck_ar_assigned_nonneg"),
        Index("ix_assignment_results_courier_date", "courier_id", "date"),
        Index("ix_assignment_results_zone_date", "zone_id", "date"),
    )
