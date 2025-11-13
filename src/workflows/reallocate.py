import os, csv
from typing import Optional, Tuple
import pandas as pd
from dotenv import load_dotenv

from .. import data_processor as dp
from ..shimbox_client import (
    admin_login, driver_login, get_approved_drivers, fetch_all_products_for_driver,
    list_unassigned_products, product_create, delete_active_assignments,
    assign_from_pool, get_driver_health_postwork
)

load_dotenv()

DISTRICT = os.getenv("DISTRICT_FILTER") or None
TARGET_NAME = os.getenv("TARGET_DRIVER_NAME", "")
DO_CREATE = os.getenv("DO_CREATE_FROM_EXCEL","true").lower() == "true"
DO_RESET = os.getenv("DO_RESET_AND_REALLOC","true").lower() == "true"
EXCEL_PATH = os.getenv("EXCEL_PATH","./data/orders.xlsx")

def _parse_role_cap(s: str) -> Tuple[int,int,int]:
    try:
        a,b,c = [int(x.strip()) for x in s.split(",")]
        return a,b,c
    except:
        return (10,20,30)
ROLE_CAP = _parse_role_cap(os.getenv("ROLE_BASE_CAP","10,20,30"))

def career_to_cap(career: str) -> int:
    base = {"초보자": ROLE_CAP[0], "경력자": ROLE_CAP[1], "숙련자": ROLE_CAP[2]}
    return base.get(career, ROLE_CAP[1])

def decide_target_count(career: Optional[str], finish3: Optional[int]) -> int:
    base = career_to_cap(career or "경력자")
    if finish3 == 1:     # 더 많이
        return int(round(base * 1.6))
    if finish3 == -1:    # 적게
        return max(1, int(round(base * 0.6)))
    return base          # 기본

def pick_target_driver(df: pd.DataFrame, target_name: str) -> Optional[pd.Series]:
    if not target_name or df.empty:
        return None
    cand = df[df["name"] == target_name]
    return cand.iloc[0] if not cand.empty else None

def create_unassigned_from_excel(admin_token: Optional[str], district: Optional[str]) -> int:
    df = dp.load_orders_from_excel(EXCEL_PATH)
    if district:
        df = df[df["address"].astype(str).str.contains(district)]
    print(f"[OK] 엑셀 로드(필터 적용): {len(df)}건")

    created = 0
    logs = []
    for _, row in df.iterrows():
        payload = {
            "name": row["product_name"],
            "qty": int(row["qty"]),
            "recipient": row["name"],
            "phone": row["phone"],
            "postal": row["postal"],
            "address": row["address"],
            "assigned": False,
            "district": district,
        }
        try:
            product_create(admin_token, payload)
            created += 1
            logs.append({"product_name": row["product_name"], "qty": int(row["qty"]), "status": "created"})
        except Exception as e:
            logs.append({"product_name": row["product_name"], "qty": int(row["qty"]), "status": f"fail:{e}"})

    if logs:
        with open("created_products.csv","w",newline="",encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["product_name","qty","status"])
            w.writeheader(); w.writerows(logs)
        print(f"[OK] 생성 성공: {created}건 → created_products.csv")
    return created

def main():
    print("[DEBUG] 관리자 로그인…")
    admin_token = admin_login(os.getenv("API_USERNAME","admin"), os.getenv("API_PASSWORD","password"))
    if admin_token: print("[OK] 관리자 토큰 OK")
    else: print("[WARN] 관리자 토큰 없음(비로그인 모드)")

    if DO_CREATE:
        print("[STEP] 엑셀 기반 미할당 상품 생성")
        create_unassigned_from_excel(admin_token, DISTRICT)

    drivers = get_approved_drivers(admin_token)
    target = pick_target_driver(drivers, TARGET_NAME)
    if target is None:
        print("[ERROR] 대상 기사를 찾을 수 없습니다."); return
    driver_id = int(target["driverId"])
    career = str(target.get("career") or "경력자")

    print("===== 타깃 기사 =====")
    print(f"driverId={driver_id}, name={target['name']}, career={career}")

    h = get_driver_health_postwork(admin_token, driver_id) or {}
    finish3 = h.get("finish3")   # 1=더 많이, -1=적게, 기타=기본
    desired = decide_target_count(career, finish3)
    if finish3 == 1: print("건강/설문: 더 많이 → 1.6배 반영")
    elif finish3 == -1: print("건강/설문: 적게 → 0.6배 반영")
    else: print("건강/설문 없음 → 베이스 적용")
    print(f"desired={desired}")

    if DISTRICT:
        print(f"지역 필터: '{DISTRICT}'")

    if DO_RESET:
        print("[STEP] 활성(대기/시작) 배정 삭제 시도")
        driver_token = driver_login(os.getenv("DRIVER_PHONE",""), os.getenv("DRIVER_PASSWORD",""))
        ok = False
        try:
            res = delete_active_assignments(driver_token)
            ok = (res.get("ok") is True) or (res.get("status") in (200,204))
        except Exception:
            ok = False
        if ok:
            print("[INFO] 삭제 완료")
        else:
            print("[WARN] 삭제 실패 → SOFT RESET(추가 배정만 수행)")

    try:
        cur = fetch_all_products_for_driver(admin_token, driver_id)
        active_cnt = int((cur["status"].isin(["waiting","started"])).sum()) if not cur.empty else 0
    except Exception:
        active_cnt = 0

    lack = max(0, desired - active_cnt)
    if lack <= 0:
        print(f"[FINAL] 활성(대기+시작)={active_cnt}, 목표={desired}")
        print("===== 완료 =====")
        return

    print(f"[STEP] 자동 배정: 부족={lack}, 지역={DISTRICT or 'ALL'}")
    res = assign_from_pool(admin_token, driver_id=driver_id, limit=lack, district=DISTRICT)
    assigned = int(res.get("assigned", lack)) if isinstance(res, dict) else lack
    print(f"[RESULT] 추가 배정 완료: {assigned}/{lack}건")

    try:
        cur = fetch_all_products_for_driver(admin_token, driver_id)
        active_cnt = int((cur["status"].isin(["waiting","started"])).sum()) if not cur.empty else desired
    except Exception:
        pass
    print(f"[FINAL] 활성(대기+시작)={active_cnt}, 목표={desired}")
    print("===== 완료 =====")

if __name__ == "__main__":
    main()
