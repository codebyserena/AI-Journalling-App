from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import List
import jwt
from .predict_emotions import assess_safety, explain_prediction, predict_emotions
from .db import SessionLocal, Prediction, PredictionFeedback, User
from .aggregation import get_daily_aggregation, get_monthly_aggregation
from .emotions import EMOTION_LABELS
import bcrypt
import os
import uuid

app = FastAPI(title="Emotion Detection API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:5173", "http://127.0.0.1:8080", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"

class TextRequest(BaseModel):
    text: str

class UserCreate(BaseModel):
    email: str
    password: str
    name: str

class UserLogin(BaseModel):
    email: str
    password: str

class TopicRequest(BaseModel):
    texts: List[str]

class FeedbackRequest(BaseModel):
    rating: str
    corrected_emotion: str | None = None
    note: str | None = None

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=7)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except Exception:
        return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: SessionLocal = Depends(get_db)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        return user
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

class TokenVerify(BaseModel):
    token: str

@app.post("/api/auth/verify")
def verify_auth_token(request: TokenVerify, db: SessionLocal = Depends(get_db)):
    token = request.token  # Now it's properly extracted from JSON body
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        return {
            "user": {
                "id": str(user.id),
                "email": user.email,
                "name": user.name
            },
            "valid": True
        }
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@app.get("/")
def root():
    return {"message": "Emotion Detection API is running!"}

@app.post("/api/auth/signup")
def signup(user_data: UserCreate, db: SessionLocal = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Hash password
    hashed_password = bcrypt.hashpw(user_data.password.encode('utf-8'), bcrypt.gensalt())
    
    # Create user
    user = User(
        id=str(uuid.uuid4()),
        email=user_data.email,
        password=hashed_password.decode('utf-8'),
        name=user_data.name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # Generate token
    token = create_access_token({"sub": str(user.id)})
    
    return {
        "user": {
            "id": str(user.id),
            "email": user.email,
            "name": user.name
        },
        "token": token
    }

@app.post("/api/auth/login")
def login(login_data: UserLogin, db: SessionLocal = Depends(get_db)):
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    if not bcrypt.checkpw(login_data.password.encode('utf-8'), user.password.encode('utf-8')):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    # Generate token
    token = create_access_token({"sub": str(user.id)})
    
    return {
        "user": {
            "id": str(user.id),
            "email": user.email,
            "name": user.name
        },
        "token": token
    }

@app.post("/api/predict")
def predict(request: TextRequest, current_user: User = Depends(get_current_user), db: SessionLocal = Depends(get_db)):
    predictions = predict_emotions(request.text)
    
    # Create prediction record
    pred_row = Prediction(
        text=request.text,
        user_id=current_user.id
    )
    
    # Set emotion probabilities
    for emotion_data in predictions["emotions"]:
        emotion_name = emotion_data["emotion"]
        probability = emotion_data["probability"]
        
        # Map to column name
        if hasattr(pred_row, emotion_name):
            setattr(pred_row, emotion_name, probability)
    
    db.add(pred_row)
    db.commit()
    db.refresh(pred_row)
    
    return {
        **predictions,
        "predictions": predictions,
        "entry_id": pred_row.id
    }

@app.get("/api/entries")
def get_user_entries(current_user: User = Depends(get_current_user), db: SessionLocal = Depends(get_db)):
    entries = db.query(Prediction).filter(Prediction.user_id == current_user.id).order_by(Prediction.timestamp.desc()).all()
    
    formatted_entries = []
    for entry in entries:
        # Extract emotion data
        emotions_data = []
        for emotion in EMOTION_LABELS:
            prob = getattr(entry, emotion)
            if prob is not None and prob > 0:
                emotions_data.append({
                    "emotion": emotion,
                    "probability": float(prob)
                })
        
        # Find primary emotion
        if emotions_data:
            primary_emotion = max(emotions_data, key=lambda x: x["probability"])["emotion"]
        else:
            primary_emotion = "neutral"
        
        # Calculate valence and arousal
        valence = calculate_entry_valence(emotions_data)
        arousal = calculate_entry_arousal(emotions_data)
        
        # Extract keywords
        keywords = extract_keywords(entry.text)
        selected_for_explanation = [
            (EMOTION_LABELS.index(item["emotion"]), item["probability"])
            for item in sorted(emotions_data, key=lambda item: item["probability"], reverse=True)[:4]
            if item["emotion"] in EMOTION_LABELS
        ]
        
        formatted_entries.append({
            "id": str(entry.id),
            "date": entry.timestamp.isoformat() if entry.timestamp else datetime.utcnow().isoformat(),
            "content": entry.text,
            "preview": entry.text[:100] + "..." if len(entry.text) > 100 else entry.text,
            "primaryEmotion": primary_emotion,
            "emotions": emotions_data,
            "keywords": keywords,
            "explanations": explain_prediction(entry.text, selected_for_explanation),
            "safety": assess_safety(entry.text, selected_for_explanation),
            "valence": valence,
            "arousal": arousal,
            "analyzed": True
        })
    
    return {
        "entries": formatted_entries
    }

@app.delete("/api/entries/{entry_id}")
def delete_entry(entry_id: int, current_user: User = Depends(get_current_user), db: SessionLocal = Depends(get_db)):
    entry = db.query(Prediction).filter(
        Prediction.id == entry_id,
        Prediction.user_id == current_user.id
    ).first()

    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Entry not found"
        )

    db.delete(entry)
    db.commit()
    return {"deleted": True, "entry_id": entry_id}

@app.post("/api/entries/{entry_id}/feedback")
def submit_entry_feedback(
    entry_id: int,
    request: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: SessionLocal = Depends(get_db),
):
    entry = db.query(Prediction).filter(
        Prediction.id == entry_id,
        Prediction.user_id == current_user.id
    ).first()

    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Entry not found"
        )

    allowed_ratings = {"right", "wrong", "unsure"}
    if request.rating not in allowed_ratings:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"rating must be one of: {', '.join(sorted(allowed_ratings))}"
        )

    if request.corrected_emotion and request.corrected_emotion not in EMOTION_LABELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unknown corrected emotion"
        )

    feedback = PredictionFeedback(
        prediction_id=entry.id,
        user_id=current_user.id,
        rating=request.rating,
        corrected_emotion=request.corrected_emotion,
        note=request.note,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return {"saved": True, "feedback_id": feedback.id}

def calculate_entry_valence(emotions_data):
    """Calculate valence from emotions"""
    positive_emotions = ["joy", "love", "gratitude", "amusement", "excitement", "optimism", "admiration"]
    negative_emotions = ["sadness", "anger", "fear", "grief", "disgust", "disappointment", "remorse"]
    
    positive_score = sum(e["probability"] for e in emotions_data if e["emotion"] in positive_emotions)
    negative_score = sum(e["probability"] for e in emotions_data if e["emotion"] in negative_emotions)
    
    total = positive_score + negative_score
    if total == 0:
        return 0.0
    return (positive_score - negative_score) / total

def calculate_entry_arousal(emotions_data):
    """Calculate arousal from emotions"""
    high_arousal = ["excitement", "anger", "fear", "nervousness", "surprise"]
    low_arousal = ["relief", "sadness", "grief", "neutral"]
    
    high_score = sum(e["probability"] for e in emotions_data if e["emotion"] in high_arousal)
    low_score = sum(e["probability"] for e in emotions_data if e["emotion"] in low_arousal)
    
    total = high_score + low_score
    if total == 0:
        return 0.5
    return high_score / total

def extract_keywords(text: str):
    """Extract simple keywords from text"""
    import re
    from collections import Counter
    
    words = re.findall(r'\b[a-z]{4,}\b', text.lower())
    stop_words = {'that', 'with', 'have', 'this', 'from', 'they', 'what', 
                  'when', 'were', 'their', 'will', 'would', 'there', 'which',
                  'about', 'could', 'should', 'your', 'some', 'than', 'them'}
    
    words = [w for w in words if w not in stop_words]
    word_counts = Counter(words)
    
    return [word for word, _ in word_counts.most_common(4)]

@app.get("/api/stats/daily")
def daily_stats(current_user: User = Depends(get_current_user)):
    return get_daily_aggregation(current_user.id)

@app.get("/api/stats/monthly")
def monthly_stats(current_user: User = Depends(get_current_user)):
    return get_monthly_aggregation(current_user.id)

@app.post("/api/topics")
def get_topics(request: TopicRequest, current_user: User = Depends(get_current_user)):
    # Simple topic extraction
    topics = []
    for i, text in enumerate(request.texts[:4]):
        keywords = extract_keywords(text)
        topics.append({
            "id": f"topic_{i}",
            "name": f"{keywords[0].capitalize()} & Reflection" if keywords else "General Reflection",
            "keywords": keywords[:3] if keywords else ["reflection", "thoughts", "feelings"],
            "entryCount": 1,
            "primaryEmotion": "calm"
        })
    
    return {"topics": topics}

# Public test endpoint (no auth required)
@app.post("/api/test/predict")
def test_predict(request: TextRequest, db: SessionLocal = Depends(get_db)):
    predictions = predict_emotions(request.text)
    
    # Create prediction record without user
    pred_row = Prediction(
        text=request.text,
        user_id=None
    )
    
    for emotion_data in predictions["emotions"]:
        emotion_name = emotion_data["emotion"]
        probability = emotion_data["probability"]
        
        if hasattr(pred_row, emotion_name):
            setattr(pred_row, emotion_name, probability)
    
    db.add(pred_row)
    db.commit()
    db.refresh(pred_row)
    
    return {
        **predictions,
        "predictions": predictions,
        "entry_id": pred_row.id,
        "message": "This is a test endpoint - no authentication required"
    }
