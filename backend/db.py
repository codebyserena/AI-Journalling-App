from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import uuid

DATABASE_URL = "sqlite:///./predictions.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship
    predictions = relationship("Prediction", back_populates="user")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)  # Allow null for now
    
    # Each emotion as float column
    admiration = Column(Float, default=0.0)
    amusement = Column(Float, default=0.0)
    anger = Column(Float, default=0.0)
    annoyance = Column(Float, default=0.0)
    approval = Column(Float, default=0.0)
    caring = Column(Float, default=0.0)
    confusion = Column(Float, default=0.0)
    curiosity = Column(Float, default=0.0)
    desire = Column(Float, default=0.0)
    disappointment = Column(Float, default=0.0)
    disapproval = Column(Float, default=0.0)
    disgust = Column(Float, default=0.0)
    embarrassment = Column(Float, default=0.0)
    excitement = Column(Float, default=0.0)
    fear = Column(Float, default=0.0)
    gratitude = Column(Float, default=0.0)
    grief = Column(Float, default=0.0)
    joy = Column(Float, default=0.0)
    love = Column(Float, default=0.0)
    nervousness = Column(Float, default=0.0)
    optimism = Column(Float, default=0.0)
    pride = Column(Float, default=0.0)
    realization = Column(Float, default=0.0)
    relief = Column(Float, default=0.0)
    remorse = Column(Float, default=0.0)
    sadness = Column(Float, default=0.0)
    surprise = Column(Float, default=0.0)
    neutral = Column(Float, default=0.0)
    
    # Relationship
    user = relationship("User", back_populates="predictions")

class PredictionFeedback(Base):
    __tablename__ = "prediction_feedback"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    rating = Column(String, nullable=False)
    corrected_emotion = Column(String, nullable=True)
    note = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Create tables if not exist
Base.metadata.create_all(bind=engine)
