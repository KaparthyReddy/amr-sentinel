from pydantic import BaseModel, Field, field_validator

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


class AnalyzeRequest(BaseModel):
    sequence: str = Field(..., min_length=20, description="Protein amino acid sequence")

    @field_validator("sequence")
    @classmethod
    def validate_amino_acids(cls, value: str) -> str:
        cleaned = value.strip().upper()
        invalid_chars = set(cleaned) - VALID_AMINO_ACIDS
        if invalid_chars:
            raise ValueError(
                f"Sequence contains characters that aren't standard amino acids: {sorted(invalid_chars)}"
            )
        return cleaned


class AnalyzeResponse(BaseModel):
    predicted_mechanism: str
    confidence: float
    novelty_distance: float
    is_novel: bool
    sequence_length: int


class SurveillanceSummaryResponse(BaseModel):
    window_minutes: int
    total_analyzed: int
    by_mechanism: dict[str, int]
    novel_count: int
    novel_rate: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    esm_model: str
    known_mechanisms: list[str]
