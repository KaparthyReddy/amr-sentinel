from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.models.embedding_service import EmbeddingService
from app.models.classifier_service import ClassifierService
from app.models.novelty_detector import NoveltyDetector
from app.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ESM2 loading is the slow part of startup (a few seconds) - done once
    # here rather than per-request.
    app.state.embedding_service = EmbeddingService()
    app.state.classifier_service = ClassifierService()
    app.state.novelty_detector = NoveltyDetector()
    yield


app = FastAPI(
    title="AMR Sentinel",
    description="AI-powered antimicrobial resistance surveillance using pretrained protein embeddings",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
