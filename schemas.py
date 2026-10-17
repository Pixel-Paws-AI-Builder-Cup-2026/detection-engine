from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class AnimalCreate(BaseModel):
    name: Optional[str] = "Unknown"
    species: str
    description: Optional[str] = None
    embedding: List[float]

class AnimalResponse(BaseModel):
    id: int
    name: Optional[str]
    species: str
    description: Optional[str]
    severity_level: str
    severity_score: int
    triage_notes: Optional[str]
    status: str
    claimed_by_volunteer_name: Optional[str]
    claimed_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True

class ClaimCaseRequest(BaseModel):
    volunteer_name: str

class LostPetCreate(BaseModel):
    pet_name: str
    owner_contact: str
    embedding: List[float]

class ClinicResponse(BaseModel):
    id: int
    clinic_name: str
    address: Optional[str]
    phone: Optional[str]
    latitude: float
    longitude: float

    class Config:
        from_attributes = True

class NearestClinicRequest(BaseModel):
    latitude: float
    longitude: float
    limit: Optional[int] = 3

class GeminiPromptRequest(BaseModel):
    prompt: str