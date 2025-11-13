# src/load_products_from_excel.py
from __future__ import annotations

from src.api.client import admin_login
from src.api.products import create_unassigned_from_excel
from src.config import DISTRICT_FILTER


def main() -> None:
    # 1) 관리자 로그인
    admin_token = admin_login()
    print("[INFO] 관리자 로그인 성공")

    # 2) 엑셀 → /product/create 로 미할당 상품 생성
    print(f"[STEP] 엑셀 → 미할당 상품 생성 (district={DISTRICT_FILTER})")
    success, fail = create_unassigned_from_excel(
        admin_token,
        district=DISTRICT_FILTER,
    )

    print(
        f"[DONE] 엑셀 기반 상품 생성 완료: "
        f"성공={success}건, 실패={fail}건"
    )


if __name__ == "__main__":
    main()
