from pydantic import BaseModel
from typing import List


# Request schema
class CaseRequest(BaseModel):
    symptoms: str
    doctor_notes: str


# Similar case schema
class SimilarCase(BaseModel):
    case_id: str
    similarity_score: float


# Response schema
class CaseResponse(BaseModel):
    similar_cases: list
    symptoms: str
    treatment: str
    similarity_score: str