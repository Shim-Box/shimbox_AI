import pandas as pd
from tqdm import tqdm

from data_utils.api_client import create_product
from utils.logger import get_logger

logger = get_logger("init_products")

EXCEL_PATH = "data/product_dummy.xlsx"

def run():
    logger.info(f"엑셀 파일 읽는 중... {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)

    # 엑셀 컬럼명이 다르면 여기 매핑 수정
    expected_cols = [
        "productName",
        "recipientName",
        "recipientPhoneNumber",
        "address",
        "detailAddress",
        "postalCode",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"엑셀에 필요한 컬럼이 없습니다: {missing}")

    logger.info(f"총 {len(df)}개의 상품을 생성합니다.")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        payload = {
            "productName": row["productName"],
            "recipientName": row["recipientName"],
            "recipientPhoneNumber": str(row["recipientPhoneNumber"]),
            "address": row["address"],
            "detailAddress": row["detailAddress"],
            "postalCode": str(row["postalCode"]),
        }

        res = create_product(payload)
        logger.info(f"상품 생성 결과: {res.get('statusCode')} - {res.get('message')}")

    logger.info("✅ 초기 상품 생성 완료")

if __name__ == "__main__":
    run()
