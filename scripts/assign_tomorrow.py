from data_utils.api_client import (
    get_approved_drivers,
    get_driver_profile,
    get_driver_health,
    get_unassigned_products,
    assign_products,
)
from ml.inference import predict_capacity
from utils.logger import get_logger

logger = get_logger("assign_tomorrow")


def fetch_all_approved():
    """승인된 기사 목록 전체 조회"""
    page = 1
    size = 50
    drivers = []

    while True:
        res = get_approved_drivers(page=page, size=size)
        data = res.get("data", {})
        drivers += data.get("data", [])

        if page >= data.get("totalPages", 1):
            break
        page += 1

    return drivers


def extract_gu(address: str) -> str:
    """주소에서 'OO구' 추출"""
    if not address:
        return "기타"
    idx = address.find("구")
    if idx == -1:
        return "기타"
    start = address.rfind(" ", 0, idx)
    start = 0 if start == -1 else start + 1
    return address[start : idx + 1].strip() or "기타"


def build_region_product_map(products):
    """지역별 상품 묶기"""
    region_map = {}
    for p in products:
        gu = extract_gu(p.get("address", ""))
        region_map.setdefault(gu, []).append(p)
    return region_map


def get_driver_regions(d: dict, profile: dict) -> list:
    """기사 담당 지역 리스트"""
    regions = []

    # /admin/approved 응답의 regions
    if isinstance(d.get("regions"), list):
        regions += d["regions"]

    # profile의 regions
    if isinstance(profile.get("regions"), list):
        for r in profile["regions"]:
            if r not in regions:
                regions.append(r)

    # residence 에서도 'OO구' 추출해서 추가
    residence = profile.get("residence") or d.get("residence")
    if residence:
        gu = extract_gu(residence)
        if gu != "기타" and gu not in regions:
            regions.append(gu)

    return regions


def run_assignment():
    logger.info("기사·상품 조회 시작")

    drivers = fetch_all_approved()
    products_res = get_unassigned_products()
    all_products = products_res.get("data", [])

    region_product_map = build_region_product_map(all_products)

    for d in drivers:
        driver_id = d.get("driverId")
        if not driver_id:
            continue

        profile = get_driver_profile(driver_id).get("data", {}) or {}
        health = get_driver_health(driver_id).get("data", {}) or {}

        if "career" not in profile and "career" in d:
            profile["career"] = d["career"]

        driver_regions = get_driver_regions(d, profile)
        logger.info(f"{driver_id} 담당 지역: {driver_regions}")

        if not driver_regions:
            logger.warning(f"{driver_id}: 지역 정보 없음")
            continue

        cap = predict_capacity(profile, health)

        selected = []
        for region in driver_regions:
            region_list = region_product_map.get(region, [])
            while region_list and len(selected) < cap:
                selected.append(region_list.pop(0))
            if len(selected) >= cap:
                break

        if not selected:
            logger.warning(f"{driver_id}: 배정할 상품 없음")
            continue

        product_ids = [p["productId"] for p in selected]

        assign_products(driver_id, product_ids)
        logger.info(f"{driver_id} → {len(product_ids)}개 배정 완료")

    logger.info("내일 배정 완료")


if __name__ == "__main__":
    run_assignment()
