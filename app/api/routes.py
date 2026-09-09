import hashlib

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from app.db.database import get_db
from app.db.models import SurveillanceLog
from app.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse, SurveillanceSummaryResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    classifier_service = request.app.state.classifier_service
    return HealthResponse(
        status="UP",
        model_loaded=True,
        esm_model=request.app.state.embedding_service.model_name,
        known_mechanisms=classifier_service.known_mechanisms,
    )


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, body: AnalyzeRequest, db: Session = Depends(get_db)):
    embedding_service = request.app.state.embedding_service
    classifier_service = request.app.state.classifier_service
    novelty_detector = request.app.state.novelty_detector

    embedding = embedding_service.embed(body.sequence)
    predicted_mechanism, confidence = classifier_service.predict(embedding)
    novelty_distance, is_novel = novelty_detector.score(embedding)

    sequence_hash = hashlib.sha256(body.sequence.encode()).hexdigest()

    log_entry = SurveillanceLog(
        sequence_hash=sequence_hash,
        sequence_length=len(body.sequence),
        predicted_mechanism=predicted_mechanism,
        confidence=confidence,
        novelty_distance=novelty_distance,
        is_novel=is_novel,
    )
    db.add(log_entry)
    db.commit()

    return AnalyzeResponse(
        predicted_mechanism=predicted_mechanism,
        confidence=round(confidence, 4),
        novelty_distance=round(novelty_distance, 4),
        is_novel=is_novel,
        sequence_length=len(body.sequence),
    )


@router.get("/surveillance/summary", response_model=SurveillanceSummaryResponse)
async def surveillance_summary(db: Session = Depends(get_db), window_minutes: int = 1440):
    cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
    rows = db.query(SurveillanceLog).filter(SurveillanceLog.created_at >= cutoff).all()

    by_mechanism: dict[str, int] = {}
    novel_count = 0
    for row in rows:
        by_mechanism[row.predicted_mechanism] = by_mechanism.get(row.predicted_mechanism, 0) + 1
        if row.is_novel:
            novel_count += 1

    total = len(rows)
    return SurveillanceSummaryResponse(
        window_minutes=window_minutes,
        total_analyzed=total,
        by_mechanism=by_mechanism,
        novel_count=novel_count,
        novel_rate=round(novel_count / total, 4) if total > 0 else 0.0,
    )
