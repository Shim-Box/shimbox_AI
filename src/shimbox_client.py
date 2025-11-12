import os, requests, pandas as pd
from typing import Optional, Dict, Any
from dotenv import load_dotenv
load_dotenv()

BASE = os.getenv("API_BASE_URL", "http://116.39.208.72:26443").rstrip("/")
TIMEOUT = int(os.getenv("API_TIMEOUT", "8"))

ADMIN_LOGIN_EP = os.getenv("ADMIN_LOGIN_ENDPOINT", "").strip()
DRIVER_LOGIN_EP = os.getenv("DRIVER_LOGIN_ENDPOINT", "").strip()
APPROVED_DRIVERS_EP = os.getenv("APPROVED_DRIVERS_ENDPOINT", "").strip()
PRODUCTS_QUERY_EP = os.getenv("PRODUCTS_QUERY_ENDPOINT", "").strip()
PRODUCT_CREATE_EP = os.getenv("PRODUCT_CREATE_ENDPOINT", "").strip()
DELETE_ACTIVE_ASSIGNMENTS_EP = os.getenv("DELETE_ACTIVE_ASSIGNMENTS_ENDPOINT", "").strip()
ASSIGN_FROM_POOL_EP = os.getenv("ASSIGN_FROM_POOL_ENDPOINT", "").strip()
POSTWORK_HEALTH_EP = os.getenv("POSTWORK_HEALTH_ENDPOINT", "").strip()

def _u(ep: str) -> str:
    if not ep.startswith("/"):
        ep = "/" + ep
    return BASE + ep

def admin_login(user: str, pw: str) -> Optional[str]:
    if not ADMIN_LOGIN_EP:
        return None
    try:
        r = requests.post(_u(ADMIN_LOGIN_EP), json={"id": user, "pw": pw}, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        return j.get("token") or j.get("access_token") or j.get("jwt")
    except requests.RequestException:
        return None

def driver_login(phone: str, pw: str) -> Optional[str]:
    if not DRIVER_LOGIN_EP or not phone or not pw:
        return None
    try:
        r = requests.post(_u(DRIVER_LOGIN_EP), json={"phone": phone, "pw": pw}, timeout=TIMEOUT)
        r.raise_for_status()
        j = r.json()
        return j.get("token") or j.get("access_token") or j.get("jwt")
    except requests.RequestException:
        return None

def _ensure_col(df: pd.DataFrame, target: str, candidates: list) -> None:
    if target in df.columns:
        return
    for c in candidates:
        if c in df.columns:
            df[target] = df[c]
            return
    df[target] = pd.NA

def get_approved_drivers(admin_token: Optional[str]) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {admin_token}"} if admin_token else {}
    r = requests.get(_u(APPROVED_DRIVERS_EP), headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json().get("items") or r.json().get("data") or r.json()
    df = pd.DataFrame(data)

    _ensure_col(df, "driverId", ["id", "driver_id", "courierId"])
    _ensure_col(df, "name", ["fullName", "driverName"])
    _ensure_col(df, "career", ["skill", "careerLevel"])

    for c in ["driverId", "name", "career"]:
        if c not in df.columns:
            df[c] = pd.NA

    return df[["driverId", "name", "career"]]

def fetch_all_products_for_driver(admin_token: Optional[str], driver_id: int) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {admin_token}"} if admin_token else {}
    url = _u(PRODUCTS_QUERY_EP)
    r = requests.get(url, headers=headers, params={"driverId": driver_id}, timeout=TIMEOUT)
    r.raise_for_status()
    items = r.json().get("items") or r.json().get("data") or r.json()
    return pd.DataFrame(items)

def list_unassigned_products(admin_token: Optional[str], district: Optional[str]=None) -> pd.DataFrame:
    headers = {"Authorization": f"Bearer {admin_token}"} if admin_token else {}
    url = _u(PRODUCTS_QUERY_EP)
    params = {"assigned": "false"}
    if district:
        params["district"] = district
    r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    items = r.json().get("items") or r.json().get("data") or r.json()
    return pd.DataFrame(items)

def product_create(admin_token: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {admin_token}"} if admin_token else {}
    r = requests.post(_u(PRODUCT_CREATE_EP), headers=headers, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def delete_active_assignments(driver_token: Optional[str]) -> Dict[str, Any]:
    if not DELETE_ACTIVE_ASSIGNMENTS_EP:
        return {"ok": False, "status": 404, "body": {"message": "DELETE_ACTIVE_ASSIGNMENTS_ENDPOINT not set"}}
    headers = {"Authorization": f"Bearer {driver_token}"} if driver_token else {}
    r = requests.delete(_u(DELETE_ACTIVE_ASSIGNMENTS_EP), headers=headers, params={"status": "waiting,started"}, timeout=TIMEOUT)
    ok = r.ok
    body = {}
    try:
        body = r.json()
    except Exception:
        pass
    return {"ok": ok, "status": r.status_code, "body": body}

def assign_from_pool(admin_token: Optional[str], driver_id:int, limit:int, district: Optional[str]=None) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {admin_token}"} if admin_token else {}
    body: Dict[str, Any] = {"driverId": int(driver_id), "limit": int(limit)}
    if district:
        body["district"] = district
    r = requests.post(_u(ASSIGN_FROM_POOL_EP), headers=headers, json=body, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def get_driver_health_postwork(admin_token: Optional[str], driver_id:int) -> Dict[str, Any]:
    if not POSTWORK_HEALTH_EP:
        return {}
    headers = {"Authorization": f"Bearer {admin_token}"} if admin_token else {}
    ep = POSTWORK_HEALTH_EP.replace("{driver_id}", str(driver_id))
    try:
        r = requests.get(_u(ep), headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return {}
