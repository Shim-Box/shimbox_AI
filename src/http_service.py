from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.pipelines.assign_engine import run_assignment

app = FastAPI(
    title="Shimbox AI Service",
    description="자동 배정 엔진 HTTP 래퍼",
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

@app.post("/ai/assign/run")
def run_auto_assignment(
    create_from_excel: bool = False,
):
    result = run_assignment(create_products_from_excel=create_from_excel)

    return {
        "status": "ok",
        "total_assigned": result.get("total_assigned", 0),
        "drivers": result.get("drivers", []),
        "assignments": result.get("assignments", []),
    }
