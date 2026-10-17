from sqlalchemy import Column, Integer, String, Text, Float, DateTime, func
from pgvector.sqlalchemy import Vector
from database import Base


class AnimalProfile(Base):
    __tablename__ = "animal_profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=True, default="Unknown")
    species = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)

    # AI Embeddings & Severity Analysis
    embedding = Column(Vector(384))  # MegaDescriptor 384-dim vector
    severity_level = Column(String(20), default="UNKNOWN")  # CRITICAL, MEDIUM, LOW
    severity_score = Column(Integer, default=0)  # 1 to 10 scale
    triage_notes = Column(Text, nullable=True)  # Gemini AI summary

    # Volunteer Rescue Tracking (For Dashboard)
    status = Column(String(50), default="REPORTED")  # REPORTED, CLAIMED, RESCUED
    claimed_by_volunteer_name = Column(String(100), nullable=True)
    claimed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, server_default=func.now())


class LostPet(Base):
    __tablename__ = "lost_pets"

    id = Column(Integer, primary_key=True, index=True)
    pet_name = Column(String(100), nullable=False)
    owner_contact = Column(String(100), nullable=False)
    embedding = Column(Vector(384))
    status = Column(String(50), default="LOST")
    reported_at = Column(DateTime, server_default=func.now())


class VeterinaryClinic(Base):
    __tablename__ = "veterinary_clinics"

    id = Column(Integer, primary_key=True, index=True)
    clinic_name = Column(String(150), nullable=False)
    address = Column(Text, nullable=True)
    phone = Column(String(20), nullable=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)