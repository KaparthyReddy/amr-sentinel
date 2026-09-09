from sqlalchemy import Column, BigInteger, String, Integer, Float, Boolean, DateTime
from sqlalchemy.sql import func

from app.db.database import Base


class SurveillanceLog(Base):
    __tablename__ = "surveillance_logs"

    id = Column(BigInteger, primary_key=True)
    sequence_hash = Column(String(64), nullable=False)
    sequence_length = Column(Integer, nullable=False)
    predicted_mechanism = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    novelty_distance = Column(Float, nullable=False)
    is_novel = Column(Boolean, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
