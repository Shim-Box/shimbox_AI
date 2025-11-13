# src/api/products.py
from typing import Any, Dict, List, Optional
import re
import time
import pandas as pd

from src.api.client import get_json, post_json
from src.config import API_BASE_URL, PRODUCT_EXCEL_PATH


def list_unassigned_products(admin_token: str) -> List[Dict[str, Any]]:
    """
    GET /api/v1/admin/products/unassigned
    """
    url = f"{API_BASE_URL}/api/v1/admin/products/unassigned"
    try:
        js = get_json(url, token=admin_token, timeout=15)
        return (js or {}).get("data") or []
    except Exception as e:
        print(f"[WARN] unassigned 조회 실패: {e}")
        return []


def assign_product(
    admin_token: str,
    product_id: int,
    driver_id: int,
) -> bool:
    """
    POST /api/v1/admin/products/assign
    body: { "productId": 1, "driverId": 1 }
    """
    url = f"{API_BASE_URL}/api/v1/admin/products/assign"
    body = {"productId": int(product_id), "driverId": int(driver_id)}
    try:
        _ = post_json(url, token=admin_token, json_body=body, timeout=10)
        return True
    except Exception as e:
        print(f"[WARN] assign 실패 pid={product_id} → did={driver_id}: {e}")
        return False


def filter_products_by_district(
    products: List[Dict[str, Any]],
    district: Optional[str],
) -> List[Dict[str, Any]]:
    """
    주소 + 상세주소에 district 문자열이 포함된 상품만 필터링
    """
    if not district:
        return products

    out: List[Dict[str, Any]] = []
    for p in products:
        addr = str(p.get("address") or "")
        detail = str(p.get("detailAddress") or "")
        full = f"{addr} {detail}"
        if district in full:
            out.append(p)
    return out


# ==========================
#  엑셀 → /product/create
# ==========================

COLUMN_MAP = {
    "상품명": "productName",
    "수령인": "recipientName",
    "전화번호": "recipientPhoneNumber",
    "주소": "address",
    "상세주소": "detailAddress",
    "우편번호": "postalCode",
    # 이미 영어 컬럼이면 그대로
    "productName": "productName",
    "recipientName": "recipientName",
    "recipientPhoneNumber": "recipientPhoneNumber",
    "address": "address",
    "detailAddress": "detailAddress",
    "postalCode": "postalCode",
}

REQUIRED_FIELDS = [
    "productName",
    "recipientName",
    "recipientPhoneNumber",
    "address",
    "detailAddress",
    "postalCode",
]


def _normalize_phone(s: Any) -> str:
    s = str(s or "")
    digits = re.sub(r"[^0-9]", "", s)
    if len(digits) == 11 and digits.startswith("01"):
        return f"{digits[0:3]}-{digits[3:7]}-{digits[7:11]}"
    if len(digits) == 10 and digits.startswith("02"):
        return f"{digits[0:2]}-{digits[2:6]}-{digits[6:10]}"
    if len(digits) == 10 and digits.startswith("01"):
        return f"{digits[0:3]}-{digits[3:6]}-{digits[6:10]}"
    return s.strip()


def _normalize_postal(s: Any) -> str:
    s = re.sub(r"[^0-9]", "", str(s or ""))
    return s.zfill(5)[:5] if s else ""


def load_orders_from_excel(
    path: str | None = None,
    district: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    상품 더미 엑셀을 읽어서 주문 dict 리스트로 변환.
    path가 None이면 config.PRODUCT_EXCEL_PATH 사용.
    district가 있으면 address / detailAddress에 해당 문자열 포함된 행만 사용.
    """
    if path is None:
        path = str(PRODUCT_EXCEL_PATH)

    df = pd.read_excel(path)

    # 컬럼명 매핑
    rename = {c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP}
    df = df.rename(columns=rename)

    # 필수 컬럼 없으면 추가
    for col in REQUIRED_FIELDS:
        if col not in df.columns:
            df[col] = ""

    # 문자열 정리
    for col in ["productName", "recipientName", "address", "detailAddress"]:
        df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else "")

    df["recipientPhoneNumber"] = df["recipientPhoneNumber"].apply(_normalize_phone)
    df["postalCode"] = df["postalCode"].apply(_normalize_postal)

    # 필수 필드가 비어 있는 행 제거
    df = df[
        df["productName"].astype(bool)
        & df["recipientName"].astype(bool)
        & df["address"].astype(bool)
    ]

    # district 필터
    if district:
        before = len(df)

        def contains_district(row):
            addr = row.get("address", "")
            detail = row.get("detailAddress", "")
            return (district in str(addr)) or (district in str(detail))

        mask = df.apply(contains_district, axis=1)
        df = df[mask]
        print(f"[INFO] '{district}' 필터: {before} → {len(df)}건")

    return df[REQUIRED_FIELDS].to_dict("records")


def create_product(admin_token: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /api/v1/product/create
    body: {
      "productName": ...,
      "recipientName": ...,
      "recipientPhoneNumber": ...,
      "address": ...,
      "detailAddress": ...,
      "postalCode": ...
    }
    """
    url = f"{API_BASE_URL}/api/v1/product/create"
    return post_json(url, token=admin_token, json_body=payload, timeout=20)


def create_unassigned_from_excel(
    admin_token: str,
    *,
    excel_path: str | None = None,
    district: Optional[str] = None,
) -> tuple[int, int]:
    """
    엑셀에서 상품 읽어서 /product/create 로 모두 미할당 생성
    district 가 있으면 해당 행정구만 생성.
    return: (성공개수, 실패개수)
    """
    try:
        orders = load_orders_from_excel(excel_path, district)
    except FileNotFoundError:
        print(f"[WARN] 엑셀 파일을 찾을 수 없습니다: {excel_path or PRODUCT_EXCEL_PATH}")
        return (0, 0)

    if not orders:
        print("[WARN] 생성할 주문이 없습니다.")
        return (0, 0)

    success = 0
    fail = 0

    for i, od in enumerate(orders, start=1):
        payload = {
            "productName": od["productName"],
            "recipientName": od["recipientName"],
            "recipientPhoneNumber": od["recipientPhoneNumber"],
            "address": od["address"],
            "detailAddress": od.get("detailAddress", ""),
            "postalCode": od.get("postalCode", ""),
        }
        try:
            create_product(admin_token, payload)
            success += 1
        except Exception as e:
            print(f"[ERR] 상품 생성 실패 #{i}: {e}")
            fail += 1
        # 서버 부하 방지용 (선택)
        time.sleep(0.1)

    print(f"[RESULT] 상품 생성 성공 {success}건 / 실패 {fail}건")
    return (success, fail)
