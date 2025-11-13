# src/api/drivers.py
from typing import Any, Dict, List
import pandas as pd

from src.api.client import get_json
from src.config import API_BASE_URL


def get_approved_drivers(
    admin_token: str,
    *,
    residence: str | None = None,
    page: int = 1,
    size: int = 100,
) -> pd.DataFrame:
    """
    GET /api/v1/admin/approved
    
    승인된 기사 목록을 DataFrame으로 반환
    - residence 필터: "성북구" 등
    """
    url = f"{API_BASE_URL}/api/v1/admin/approved"
    params: Dict[str, Any] = {
        "page": page,
        "size": size,
    }
    if residence:
        params["residence"] = residence

    js = get_json(url, token=admin_token, params=params)
    data = (js or {}).get("data") or {}
    items = data.get("data") or []
    df = pd.DataFrame(items)
    # driverId 컬럼 보정
    if "driverId" not in df.columns and "userId" in df.columns:
        df["driverId"] = df["userId"]
    return df


def get_driver_health(admin_token: str, driver_id: int) -> Dict[str, Any]:
    """
    GET /api/v1/admin/driver/{driverId}/health
    """
    url = f"{API_BASE_URL}/api/v1/admin/driver/{driver_id}/health"
    try:
        js = get_json(url, token=admin_token, timeout=10)
        return (js or {}).get("data") or {}
    except Exception as e:
        print(f"[WARN] health 호출 실패 driverId={driver_id}: {e}")
        return {}
