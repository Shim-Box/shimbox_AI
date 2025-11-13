# src/pipelines/assign_engine.py
"""
전체 배정 엔진

1) 관리자 로그인
2) (옵션) 엑셀 → /product/create 로 미할당 상품 생성
3) 승인 기사 목록 가져오기 (성북구 residence)
4) 각 기사에 대해:
   - /admin/driver/{id}/health 로 퇴근 설문/건강 데이터 가져오기
   - ai.capacity_model.predict_capacity_for_driver 로 내일 추천 물량 계산
5) /admin/products/unassigned 에서 미할당 상품 목록 가져오기
   - 주소에 DISTRICT_FILTER(예: "성북구") 포함된 상품만 필터링
6) 각 기사별 추천 물량만큼 순서대로 assign
"""

from __future__ import annotations
from typing import Any, Dict, List

from src.api.client import admin_login
from src.api.drivers import get_approved_drivers, get_driver_health
from src.api.products import (
    list_unassigned_products,
    assign_product,
    filter_products_by_district,
    create_unassigned_from_excel,
)
from src.ai.capacity_model import predict_capacity_for_driver
from src.config import DISTRICT_FILTER, normalize_career


def run_assignment(
    *,
    create_products_from_excel: bool = False,
) -> Dict[str, Any]:
    """
    배정 파이프라인 실행.
    :param create_products_from_excel: True면 엑셀에서 /product/create로 미할당 상품 생성까지 수행.
    :return: {
      "drivers": [ { driverId, name, career, desired_capacity, assigned_count, health }, ... ],
      "assignments": [ { driverId, productId, address }, ... ],
      "total_assigned": int,
    }
    """
    # 0) 관리자 로그인
    admin_token = admin_login()
    print("[INFO] 관리자 로그인 성공")

    # 1) (옵션) 엑셀에서 미할당 상품 생성
    if create_products_from_excel:
        print("[STEP] 엑셀 → 미할당 상품 생성")
        create_unassigned_from_excel(admin_token, district=DISTRICT_FILTER)

    # 2) 승인 기사 목록 가져오기 (residence 기준 필터)
    print("[STEP] 승인 기사 목록 조회")
    df_drivers = get_approved_drivers(
        admin_token,
        residence=DISTRICT_FILTER,
        page=1,
        size=200,
    )

    if df_drivers.empty:
        print("[WARN] 승인된 기사가 없습니다.")
        return {"drivers": [], "assignments": [], "total_assigned": 0}

    # 3) 기사별 추천 물량 계산
    driver_recs: List[Dict[str, Any]] = []

    for _, row in df_drivers.iterrows():
        driver_id = int(row["driverId"])
        name = row.get("name") or row.get("driverName") or f"driver-{driver_id}"

        # raw career와 normalize된 career 분리
        career_raw = row.get("career")
        career = normalize_career(career_raw)

        # 건강 데이터
        health = get_driver_health(admin_token, driver_id)

        # Capacity 계산 (career_raw를 넘겨서 내부에서 normalize 가능하게)
        capacity = predict_capacity_for_driver(
            driver_id=driver_id,
            career_raw=career_raw,
            health=health,
        )

        driver_recs.append(
            {
                "driverId": driver_id,
                "name": name,
                "career": career,
                "desired_capacity": capacity,
                "health": health,
            }
        )

        finish3 = (health or {}).get("finish3")
        print(
            f"[REC] driverId={driver_id} name={name} "
            f"career={career} finish3={finish3} → 추천 물량={capacity}"
        )

    if not driver_recs:
        print("[WARN] 추천할 기사가 없습니다.")
        return {"drivers": [], "assignments": [], "total_assigned": 0}

    # 4) 미할당 상품 목록 가져오기
    print("[STEP] 미할당 상품 목록 조회")
    unassigned = list_unassigned_products(admin_token)
    if not unassigned:
        print("[WARN] 미할당 상품이 없습니다.")
        return {"drivers": driver_recs, "assignments": [], "total_assigned": 0}

    # 5) DISTRICT_FILTER(예: 성북구) 상품만 필터
    filtered = filter_products_by_district(unassigned, DISTRICT_FILTER)
    if not filtered:
        print(f"[WARN] '{DISTRICT_FILTER}' 주소를 가진 미할당 상품이 없습니다.")
        return {"drivers": driver_recs, "assignments": [], "total_assigned": 0}

    print(f"[INFO] 전체 미할당 상품 {len(unassigned)}건 중 '{DISTRICT_FILTER}' 상품 {len(filtered)}건 사용")

    product_iter = iter(filtered)
    assignments: List[Dict[str, Any]] = []

    # 6) 순서대로 기사에게 추천 개수만큼 assign
    for rec in driver_recs:
        driver_id = rec["driverId"]
        desired = int(rec["desired_capacity"])
        assigned_count = 0

        while assigned_count < desired:
            try:
                p = next(product_iter)
            except StopIteration:
                print("[INFO] 더 이상 배정할 상품이 없습니다.")
                break

            product_id = int(p["productId"])
            ok = assign_product(admin_token, product_id, driver_id)
            if ok:
                assigned_count += 1
                assignments.append(
                    {
                        "driverId": driver_id,
                        "productId": product_id,
                        "address": p.get("address"),
                    }
                )
            else:
                # assign 실패 시 그냥 다음 상품으로
                continue

        rec["assigned_count"] = assigned_count
        print(
            f"[ASSIGN] driverId={driver_id} ({rec['name']}) "
            f"추천={desired} → 실제 배정={assigned_count}"
        )

    total_assigned = sum(r.get("assigned_count", 0) for r in driver_recs)
    print(f"[FINAL] 전체 배정 완료: {total_assigned}건")

    return {
        "drivers": driver_recs,
        "assignments": assignments,
        "total_assigned": total_assigned,
    }
