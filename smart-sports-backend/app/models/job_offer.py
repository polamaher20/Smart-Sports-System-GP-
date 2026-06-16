from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.database import Base

class JobOffer(Base):
    __tablename__ = "job_offers"

    id          = Column(Integer, primary_key=True, index=True)
    club_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    club_name   = Column(String, nullable=False)
    title       = Column(String, nullable=False)
    position    = Column(String, nullable=False)
    location    = Column(String, nullable=True)
    description = Column(String, nullable=True)
    sport       = Column(String, nullable=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())

    # ── Requirements ──────────────────────────────────────────────
    age_min         = Column(Integer,  nullable=True)
    age_max         = Column(Integer,  nullable=True)
    height_cm       = Column(Integer,  nullable=True)
    experience_years= Column(Integer,  nullable=True)
    salary_min      = Column(Integer,  nullable=True)
    salary_max      = Column(Integer,  nullable=True)
    deadline        = Column(String,   nullable=True)
    # Skills stored as comma-separated string e.g. "shooting,pace,dribbling"
    required_skills = Column(String,   nullable=True)