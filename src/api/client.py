# src/api/client.py
import requests
from typing import Dict, Any
from src.config import API_BASE_URL, ADMIN_EMAIL, ADMIN_PASSWORD


def make_headers(token: str | None = None, *, json_ct: bool = True) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if json_ct:
        headers["Content-Type"] = "application/json; charset=utf-8"
    return headers


def admin_login() -> str:
    """
    ShimBox 관리자 로그인 → accessToken 반환
    """
    url = f"{API_BASE_URL}/api/v1/auth/login"
    payload = {"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    resp = requests.post(url, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("data") or {}
    token = data.get("accessToken")
    if not token:
        raise RuntimeError("관리자 로그인은 성공했으나 accessToken이 없습니다.")
    return token


def get_json(
    url: str,
    *,
    token: str | None = None,
    params: Dict[str, Any] | None = None,
    timeout: int = 15,
) -> Any:
    resp = requests.get(url, headers=make_headers(token, json_ct=False), params=params, timeout=timeout)
    resp.raise_for_status()
    if not resp.text.strip():
        return None
    return resp.json()


def post_json(
    url: str,
    *,
    token: str | None = None,
    json_body: Dict[str, Any] | None = None,
    timeout: int = 15,
) -> Any:
    resp = requests.post(url, headers=make_headers(token), json=json_body, timeout=timeout)
    resp.raise_for_status()
    if not resp.text.strip():
        return None
    return resp.json()
