import io
import json
import os
import torch
from torchvision import transforms
from PIL import Image
from datetime import datetime
from fastapi import FastAPI, Depends, HTTPException, File, UploadFile, Form, Response
from sqlalchemy.orm import Session
from typing import List, Optional
from google import genai
from google.genai import types
from contextlib import asynccontextmanager
from database import get_db, engine, Base
import models
import schemas

# Automatically create database tables if they don't exist
Base.metadata.create_all(bind=engine)

# ------------------------------------------------------------------
# MODEL STATE CONTAINER (Loaded during startup)
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# LAZY-LOAD MODEL INITIALIZATION (Prevents Cloud Run Startup Timeout)
# ------------------------------------------------------------------
ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- LOAD MODEL ON STARTUP ---
    print("🚀 Loading MegaDescriptor-L-384 weights from Hugging Face into memory...")
    import timm
    ml_resources["model"] = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True).eval()
    ml_resources["transform"] = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("✨ MegaDescriptor-L-384 loaded successfully and ready!")
    
    yield
    
    # --- CLEANUP ON SHUTDOWN ---
    ml_resources.clear()

app = FastAPI(title="Pixel and Paws Backend API", version="2.6.0", lifespan=lifespan)

# Initialize Gemini 3.8 Flash Client (Reads GEMINI_API_KEY from environment)
ai_client = genai.Client()

def generate_backend_embedding(image_bytes: bytes):
    # Pull pre-loaded model & transform directly from memory state
    print("✨ Pulling the model and transformation pipeline from memory")
    model = ml_resources["model"]
    transform_pipeline = ml_resources["transform"]
    
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor_img = transform_pipeline(image).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor_img)
    print("✨ Backend embedding generated")
    return features.squeeze(0).cpu().tolist()


@app.get("/")
def root():
    return {"message": "Pixel and Paws API is fully operational for Google Cloud AI Builder Cup!"}


# ------------------------------------------------------------------
# 1. ANIMAL / STRAY PROFILES: Upload, Embed, & Gemini Severity Triage
# ------------------------------------------------------------------
@app.post("/api/v1/animals/upload", response_model=schemas.AnimalResponse)
async def upload_animal_and_evaluate(
    name: Optional[str] = Form("Unknown"),
    species: str = Form(...),
    description: Optional[str] = Form(""),
    latitude: float = Form(...),
    longitude: float = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    image_bytes = await file.read()
    
    # 1. Generate 1536-dim visual embedding using MegaDescriptor
    vector_embedding = generate_backend_embedding(image_bytes)

    # 2. Analyze severity & triage using Gemini 2.0 Flash
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=file.content_type or "image/jpeg"
    )

    prompt = (
        f"You are a veterinary AI triage assistant. Analyze this animal photo and report notes: '{description}'. "
        "Return ONLY a raw JSON object with no markdown formatting: "
        "{"
        "  \"severity_level\": \"CRITICAL\" or \"MEDIUM\" or \"LOW\","
        "  \"severity_score\": integer between 1 and 10,"
        "  \"triage_notes\": \"brief summary of visible wounds/distress and immediate first aid steps\""
        "}"
    )

    severity_level = "LOW"
    severity_score = 1
    triage_notes = "No visible severe distress reported."

    try:
        response = ai_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[image_part, prompt],
        )
        clean_json_text = response.text.replace("```json", "").replace("```", "").strip()
        triage_data = json.loads(clean_json_text)
        
        severity_level = triage_data.get("severity_level", "MEDIUM")
        severity_score = int(triage_data.get("severity_score", 5))
        triage_notes = triage_data.get("triage_notes", "")

        print("\n--- [GEMINI TRIAGE EVALUATION REPORT] ---")
        print(f"🐾 Patient Name: {name}")
        print(f"🚨 Severity Level: {severity_level}")
        print(f"📊 Severity Score: {severity_score}/10")
        print(f"📝 Triage Notes: {triage_notes}")
        print("-------------------------------------------\n")
    except Exception as e:
        print(f"Gemini Triage Error: {e}")

    # 3. Save to Cloud SQL PostgreSQL database
    db_animal = models.AnimalProfile(
        name=name,
        species=species,
        embedding=vector_embedding,
        description=description,
        severity_level=severity_level,
        severity_score=severity_score,
        triage_notes=triage_notes,
        status="REPORTED",
        image_data=image_bytes,
        latitude=latitude,
        longitude=longitude
    )
    db.add(db_animal)
    db.commit()
    db.refresh(db_animal)

    return db_animal


# ------------------------------------------------------------------
# 2. LOST PETS & VECTOR SIMILARITY MATCHING (pgvector)
# ------------------------------------------------------------------
@app.post("/api/v1/lost-pets/report-upload")
async def report_lost_pet_upload(
    pet_name: Optional[str] = Form("Unknown"),
    owner_name: str = Form(...),
    owner_contact: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Read image bytes and generate 1536-dim embedding
    image_bytes = await file.read()
    embedding_vector = generate_backend_embedding(image_bytes)

    # 2. Save lost pet report to database
    db_pet = models.LostPet(
        pet_name=pet_name,
        owner_name=owner_name,
        owner_contact=owner_contact,
        embedding=embedding_vector,
        status="LOST",
        image_data=image_bytes
    )
    db.add(db_pet)
    db.commit()
    db.refresh(db_pet)

    # 3. Cross-match against stray/found animal profiles using pgvector cosine distance
    # This checks if civilians have already uploaded a matching animal in the rescue cases!
    potential_matches = db.query(models.AnimalProfile)\
        .filter(models.AnimalProfile.embedding.cosine_distance(embedding_vector) < 0.4)\
        .order_by(models.AnimalProfile.embedding.cosine_distance(embedding_vector))\
        .limit(3)\
        .all()

    # 4. Return the newly created lost pet info along with any potential found matches
    return {
        "message": "Lost pet reported successfully!",
        "lost_pet_id": db_pet.id,
        "potential_rescue_matches": [
            {
                "animal_id": match.id,
                "species": match.species,
                "description": match.description,
                "severity_level": match.severity_level,
                "latitude": match.latitude,
                "longitude": match.longitude,
                "last_reported_at": match.created_at.replace(microsecond=0)
            }
            for match in potential_matches
        ]
    }

    # ------------------------------------------------------------------
# LOST PETS: GET ENDPOINTS
# ------------------------------------------------------------------
@app.get("/api/v1/lost-pets/", response_model=List[schemas.LostPetResponse])
def get_all_lost_pets(
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(models.LostPet)
    if status_filter:
        query = query.filter(models.LostPet.status == status_filter)
    
    # Return newest lost pet reports first
    return query.order_by(models.LostPet.reported_at.desc()).all()


@app.get("/api/v1/lost-pets/{pet_id}", response_model=schemas.LostPetResponse)
def get_lost_pet_by_id(
    pet_id: int,
    db: Session = Depends(get_db)
):
    pet = db.query(models.LostPet).filter(models.LostPet.id == pet_id).first()
    if not pet:
        raise HTTPException(status_code=404, detail="Lost pet report not found")
    return pet

@app.get("/api/v1/lost-pets/{pet_id}/image")
def get_lost_pet_image(pet_id: int, db: Session = Depends(get_db)):
    pet = db.query(models.LostPet).filter(models.LostPet.id == pet_id).first()
    if not pet or not pet.image_data:
        raise HTTPException(status_code=404, detail="Lost pet image not found")
    
    return Response(content=pet.image_data, media_type="image/jpeg")

# ------------------------------------------------------------------
# 3. DASHBOARD & VOLUNTEER RESCUE CLAIM WORKFLOW
# ------------------------------------------------------------------
@app.get("/api/v1/dashboard/cases", response_model=List[schemas.AnimalResponse])
def get_dashboard_cases(
    status_filter: Optional[str] = None, 
    db: Session = Depends(get_db)
):
    query = db.query(models.AnimalProfile)
    if status_filter:
        query = query.filter(models.AnimalProfile.status == status_filter)
    
    # Sort critical, high-score cases to the top of the dashboard
    return query.order_by(models.AnimalProfile.severity_score.desc()).all()


@app.post("/api/v1/animals/{animal_id}/claim", response_model=schemas.AnimalResponse)
def claim_rescue_case(
    animal_id: int, 
    payload: schemas.ClaimCaseRequest, 
    db: Session = Depends(get_db)
):
    animal = db.query(models.AnimalProfile).filter(models.AnimalProfile.id == animal_id).first()
    if not animal:
        raise HTTPException(status_code=404, detail="Animal report not found")
    
    if animal.status == "CLAIMED":
        raise HTTPException(
            status_code=400, 
            detail=f"Case already claimed by volunteer {animal.claimed_by_volunteer_name}"
        )

    animal.status = "CLAIMED"
    animal.claimed_by_volunteer_name = payload.volunteer_name
    animal.claimed_at = datetime.utcnow()

    db.commit()
    db.refresh(animal)
    return animal


# ------------------------------------------------------------------
# 4. VETERINARY CLINICS (Nearest Location Lookup)
# ------------------------------------------------------------------
@app.get("/api/v1/clinics/", response_model=List[schemas.ClinicResponse])
def get_all_clinics(db: Session = Depends(get_db)):
    return db.query(models.VeterinaryClinic).all()


@app.post("/api/v1/clinics/nearest", response_model=List[schemas.ClinicResponse])
def get_nearest_clinics(payload: schemas.NearestClinicRequest, db: Session = Depends(get_db)):
    lat = payload.latitude
    lon = payload.longitude

    clinics = (
        db.query(models.VeterinaryClinic)
        .order_by(
            ((models.VeterinaryClinic.latitude - lat) * (models.VeterinaryClinic.latitude - lat) +
             (models.VeterinaryClinic.longitude - lon) * (models.VeterinaryClinic.longitude - lon))
        )
        .limit(payload.limit)
        .all()
    )
    return clinics


# ------------------------------------------------------------------
# 5. GEMINI 2.0 FLASH CHAT ASSISTANT
# ------------------------------------------------------------------
@app.post("/api/v1/assistant/chat")
def chat_with_assistant(payload: schemas.GeminiPromptRequest):
    try:
        response = ai_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=payload.prompt,
        )
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/animals/{animal_id}/image")
def get_animal_image(animal_id: int, db: Session = Depends(get_db)):
    animal = db.query(models.AnimalProfile).filter(models.AnimalProfile.id == animal_id).first()
    if not animal or not animal.image_data:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Return raw bytes with JPEG media type so the browser/frontend renders it instantly
    return Response(content=animal.image_data, media_type="image/jpeg")

# ------------------------------------------------------------------
# SERVER ENTRY POINT: Binds to $PORT for Cloud Run compatibility
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)