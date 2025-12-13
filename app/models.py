"""
Pydantic models for request/response validation
"""
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class OCRRequest(BaseModel):
    """Request model for OCR extraction"""
    document_type: str = Field(..., description="Type of document: ID_CARD, FORM, CERTIFICATE, etc.")


class FieldResult(BaseModel):
    """Result for a single extracted field"""
    field: str
    value: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class ConfidenceZone(BaseModel):
    """A single OCR confidence zone with word-level information"""
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    x: int
    y: int
    width: int
    height: int


class OCRExtractionResponse(BaseModel):
    """Response model for OCR extraction"""
    document_type: str
    raw_text: str
    structured_data: Dict[str, str]
    confidence_scores: Dict[str, float]
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    field_details: Optional[List[FieldResult]] = None
    quality_score: Optional[Dict[str, float]] = None
    confidence_zones: Optional[List[ConfidenceZone]] = None


class VerificationFieldResult(BaseModel):
    """Result for a single field verification"""
    field: str
    document_value: str
    form_value: str
    status: Literal["match", "mismatch", "unsure"]
    similarity_score: float = Field(..., ge=0.0, le=1.0)


class VerificationRequest(BaseModel):
    """Request model for data verification"""
    form_data: Dict[str, str] = Field(..., description="User-filled form data as key-value pairs")


class VerificationResponse(BaseModel):
    """Response model for data verification"""
    overall_verification_score: float = Field(..., ge=0.0, le=1.0)
    field_results: List[VerificationFieldResult]
    mismatched_fields: List[str]
    missing_fields: List[str]

