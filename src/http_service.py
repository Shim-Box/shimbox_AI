# src/http_service.py
from __future__ import annotations

from tempfile import NamedTemporaryFile
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from src.api.client import admin_login
from src.api.products import create_unassigned_from_excel
from src.pipelines.assign_engine import run_assignment
from src.config import DISTRICT_FILTER

app = FastAPI(
    title="Shimbox AI Service",
    description="엑셀 기반 미할당 상품 생성 + 자동 배정 엔진 HTTP 래퍼",
)

# CORS: 운영 + 로컬 개발 둘 다 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://shimbox.suple.cloud:26441",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ai/unassigned/upload-excel")
async def upload_unassigned_excel(
    file: UploadFile = File(...),
    district: str = Form(DISTRICT_FILTER),
) -> Dict[str, Any]:

    admin_token = admin_login()

    with NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    success, fail = create_unassigned_from_excel(
        admin_token,
        excel_path=tmp_path,
        district=district if district else None,
    )

    return {
        "status": "ok",
        "district": district,
        "success": success,
        "fail": fail,
    }


@app.post("/ai/assign/run")
def run_auto_assignment(
    create_from_excel: bool = False,
) -> Dict[str, Any]:

    result = run_assignment(create_products_from_excel=create_from_excel)

    return {
        "status": "ok",
        "total_assigned": result.get("total_assigned", 0),
        "drivers": result.get("drivers", []),
        "assignments": result.get("assignments", []),
    }
