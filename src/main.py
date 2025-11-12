from fastapi import FastAPI
from .database import Base, engine
from .api_server import router

try:
    from .ml_model import load_patchtst_model
    SHOULD_LOAD = True
except Exception:
    SHOULD_LOAD = False

def create_app() -> FastAPI:
    app = FastAPI(
        title="Distribution AI API",
        description="택배기사 피로도 기반 배차 추천 + ShimBox 워크플로 자동화",
        version="1.0.0",
    )

    @app.on_event("startup")
    def on_startup():
        Base.metadata.create_all(bind=engine)
        if SHOULD_LOAD:
            load_patchtst_model()

    app.include_router(router)

    @app.get("/", tags=["Health"])
    def health():
        return {"status": "ok"}

    return app

app = create_app()
