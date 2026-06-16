from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.database import get_db
from app.core.security import decode_token
from app.models.user import User
from app.models.job_offer import JobOffer
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/jobs", tags=["Jobs"])
security = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = db.query(User).filter(User.id == int(payload.get("sub"))).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

class JobOfferCreate(BaseModel):
    title:            str
    position:         str
    location:         Optional[str] = None
    description:      Optional[str] = None
    sport:            Optional[str] = None
    # Requirements
    age_min:          Optional[int]   = None
    age_max:          Optional[int]   = None
    height_cm:        Optional[int]   = None
    experience_years: Optional[int]   = None
    salary_min:       Optional[int]   = None
    salary_max:       Optional[int]   = None
    deadline:         Optional[str]   = None
    required_skills:  Optional[str]   = None  # comma-separated e.g. "shooting,pace"

# النادي ينزل job offer
@router.post("/")
def create_job_offer(
    data: JobOfferCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "club":
        raise HTTPException(status_code=403, detail="Clubs only")

    offer = JobOffer(
        club_id          = current_user.id,
        club_name        = current_user.full_name,
        title            = data.title,
        position         = data.position,
        location         = data.location or current_user.city,
        description      = data.description,
        sport            = data.sport or current_user.sport,
        age_min          = data.age_min,
        age_max          = data.age_max,
        height_cm        = data.height_cm,
        experience_years = data.experience_years,
        salary_min       = data.salary_min,
        salary_max       = data.salary_max,
        deadline         = data.deadline,
        required_skills  = data.required_skills,
    )
    db.add(offer)
    db.commit()
    db.refresh(offer)
    return {"message": "Job offer created ✅", "id": offer.id}

# جيب كل الـ job offers
@router.get("/")
def get_all_jobs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    offers = db.query(JobOffer).order_by(JobOffer.created_at.desc()).all()
    return [
        {
            "id":               o.id,
            "club_id":          o.club_id,
            "club_name":        o.club_name,
            "title":            o.title,
            "position":         o.position,
            "location":         o.location,
            "description":      o.description,
            "sport":            o.sport,
            "created_at":       str(o.created_at),
            "age_min":          o.age_min,
            "age_max":          o.age_max,
            "height_cm":        o.height_cm,
            "experience_years": o.experience_years,
            "salary_min":       o.salary_min,
            "salary_max":       o.salary_max,
            "deadline":         o.deadline,
            "required_skills":  o.required_skills,
        }
        for o in offers
    ]

# النادي يشوف الـ offers بتاعته
@router.get("/my-offers")
def get_my_offers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    if current_user.role != "club":
        raise HTTPException(status_code=403, detail="Clubs only")
    
    offers = db.query(JobOffer).filter(
        JobOffer.club_id == current_user.id
    ).order_by(JobOffer.created_at.desc()).all()
    
    return [
        {
            "id":          o.id,
            "title":       o.title,
            "position":    o.position,
            "location":    o.location,
            "description": o.description,
            "sport":       o.sport,
            "created_at":  str(o.created_at),
        }
        for o in offers
    ]

# النادي يحذف offer
@router.delete("/{offer_id}")
def delete_job_offer(
    offer_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    offer = db.query(JobOffer).filter(
        JobOffer.id == offer_id,
        JobOffer.club_id == current_user.id
    ).first()
    if not offer:
        raise HTTPException(status_code=404, detail="Offer not found")
    
    db.delete(offer)
    db.commit()
    return {"message": "Offer deleted ✅"}