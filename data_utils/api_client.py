import requests
from utils.env import API_BASE_URL, ADMIN_EMAIL, ADMIN_PASSWORD

session = requests.Session()

BASE_URL = f"{API_BASE_URL}/api/v1"

# -----------------------------
# 토큰 관리
# -----------------------------
_token_cache = None


def session_token():
    global _token_cache
    if _token_cache:
        return _token_cache

    payload = {
        "email": ADMIN_EMAIL,
        "password": ADMIN_PASSWORD,
    }

    res = session.post(f"{BASE_URL}/auth/login", json=payload)
    res.raise_for_status()
    data = res.json()

    if "data" not in data or "accessToken" not in data["data"]:
        raise Exception(f"❌ 관리자 로그인 실패: {data}")

    _token_cache = data["data"]["accessToken"]
    return _token_cache


def _headers():
    return {
        "Authorization": f"Bearer {session_token()}",
        "Content-Type": "application/json",
    }


# -----------------------------
# 상품 API
# -----------------------------
def create_product(product_data):
    res = session.post(
        f"{BASE_URL}/product/create",
        json=product_data,
        headers=_headers(),
    )
    res.raise_for_status()
    return res.json()


def get_unassigned_products():
    res = session.get(
        f"{BASE_URL}/admin/products/unassigned",
        headers=_headers(),
    )

    if res.status_code == 404:
        # 미할당 상품 없음
        return {"data": []}

    res.raise_for_status()
    return res.json()


def assign_products(driver_id, product_ids):
    if not isinstance(product_ids, list):
        product_ids = [product_ids]

    results = []
    for pid in product_ids:
        payload = {
            "productId": pid,
            "driverId": driver_id,
        }
        res = session.post(
            f"{BASE_URL}/admin/products/assign",
            json=payload,
            headers=_headers(),
        )
        res.raise_for_status()
        results.append(res.json())

    return results


# -----------------------------
# 기사 정보 API
# -----------------------------
def get_driver_profile(driver_id: int):
    res = session.get(
        f"{BASE_URL}/admin/driver/{driver_id}/profile",
        headers=_headers(),
    )
    res.raise_for_status()
    return res.json()


def get_driver_health(driver_id: int):
    res = session.get(
        f"{BASE_URL}/admin/driver/{driver_id}/health",
        headers=_headers(),
    )

    # 건강 데이터 없음(첫 배정 등)
    if res.status_code == 404:
        return {"data": {}}

    res.raise_for_status()
    return res.json()


def get_approved_drivers(page: int = 1, size: int = 100):
    params = {"page": page, "size": size}
    res = session.get(
        f"{BASE_URL}/admin/approved",
        params=params,
        headers=_headers(),
    )
    res.raise_for_status()
    return res.json()
