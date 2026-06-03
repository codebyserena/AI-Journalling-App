from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
from .emotions import EMOTION_LABELS

MODEL_PATH = os.getenv("EMOTION_MODEL_PATH", "./backend/emotion_model")
DEFAULT_THRESHOLD = 0.25
TOP_K = 6

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
except Exception:
    # Fallback to a default model if custom one doesn't exist
    MODEL_PATH = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=28)
    model.eval()

labels = [
    model.config.id2label[index].lower()
    for index in range(len(model.config.id2label))
] if getattr(model.config, "id2label", None) and len(model.config.id2label) == model.config.num_labels else EMOTION_LABELS

EXPLANATION_TERMS = {
    "admiration": {"admire", "respect", "impressive", "inspiring", "brilliant", "capable"},
    "amusement": {"funny", "laugh", "hilarious", "joke", "smile"},
    "anger": {"angry", "furious", "rage", "mad", "frustrated"},
    "annoyance": {"annoyed", "irritated", "interrupted", "bothered", "frustrating"},
    "approval": {"agree", "approved", "right", "valid", "good"},
    "caring": {"care", "support", "kind", "gentle", "check in"},
    "confusion": {"confused", "unclear", "lost", "unsure", "wondering"},
    "curiosity": {"curious", "wonder", "question", "interested", "learn"},
    "desire": {"want", "wish", "need", "hope", "longing"},
    "disappointment": {"disappointed", "let down", "failed", "unfortunate", "expected"},
    "disapproval": {"wrong", "disagree", "bad", "unacceptable", "dismissed"},
    "disgust": {"disgusting", "gross", "awful", "repulsive", "nasty"},
    "embarrassment": {"embarrassed", "awkward", "ashamed", "humiliated"},
    "excitement": {"excited", "thrilled", "amazing", "can't wait", "energized"},
    "fear": {"afraid", "scared", "terrified", "panic", "could go wrong"},
    "gratitude": {"thanks", "thankful", "grateful", "appreciate", "kindness"},
    "grief": {"grief", "mourning", "loss", "heartbroken", "miss"},
    "joy": {"happy", "joy", "delighted", "gentle", "rested", "pride"},
    "love": {"love", "beloved", "affection", "adore", "cherish"},
    "nervousness": {"nervous", "anxious", "worried", "restless", "hard to focus"},
    "optimism": {"optimistic", "hopeful", "better", "positive", "forward"},
    "pride": {"proud", "accomplished", "achievement", "capable", "progress"},
    "realization": {"realized", "noticed", "understood", "learned", "reminded"},
    "relief": {"relieved", "finally", "safe", "easier", "rested"},
    "remorse": {"sorry", "regret", "guilty", "apologize"},
    "sadness": {"sad", "cry", "hurt", "lonely", "exhausted"},
    "surprise": {"surprised", "unexpected", "shocked", "wow"},
    "neutral": {"okay", "fine", "normal", "ordinary"},
}

DISTRESS_TERMS = {
    "crisis", "hopeless", "self-harm", "self harm", "suicide", "suicidal",
    "kill myself", "end my life", "can't go on", "cannot go on",
}

def predict_emotions(text: str):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.sigmoid(outputs.logits)
        
        probabilities = probs.squeeze().tolist()
        if isinstance(probabilities, float):
            probabilities = [probabilities]
        
        ranked = sorted(
            enumerate(probabilities),
            key=lambda item: item[1],
            reverse=True
        )
        thresholded = [item for item in ranked if item[1] >= DEFAULT_THRESHOLD]
        selected = (thresholded or ranked)[:TOP_K]
        top_emotion_idx, top_probability = ranked[0]
        
        return {
            "label": labels[top_emotion_idx],
            "probability": float(top_probability),
            "emotions": [
                {
                    "emotion": labels[idx],
                    "probability": float(prob)
                }
                for idx, prob in selected
            ],
            "explanations": explain_prediction(text, selected),
            "safety": assess_safety(text, selected),
            "valence": calculate_valence(probabilities),
            "arousal": calculate_arousal(probabilities)
        }
    except Exception as e:
        print(f"Error in emotion prediction: {e}")
        # Return mock data for testing
        return get_mock_emotions(text)

def calculate_valence(probabilities):
    """Calculate emotional valence (-1 to 1)"""
    positive_indices = [labels.index(e) for e in ["joy", "love", "gratitude", "amusement", "excitement", "optimism", "admiration"] if e in labels]
    negative_indices = [labels.index(e) for e in ["sadness", "anger", "fear", "grief", "disgust", "disappointment", "remorse"] if e in labels]
    
    positive = sum(probabilities[i] for i in positive_indices if i < len(probabilities))
    negative = sum(probabilities[i] for i in negative_indices if i < len(probabilities))
    
    total = positive + negative
    if total == 0:
        return 0.0
    return (positive - negative) / total

def explain_prediction(text: str, selected: list[tuple[int, float]]):
    """Return simple phrase matches that can help users inspect predictions."""
    text_lower = text.lower()
    explanations = []
    for emotion_index, probability in selected[:4]:
        emotion = labels[emotion_index]
        matches = [
            term
            for term in sorted(EXPLANATION_TERMS.get(emotion, set()), key=len, reverse=True)
            if term in text_lower
        ][:3]
        if matches:
            explanations.append({
                "emotion": emotion,
                "probability": float(probability),
                "matched_terms": matches,
            })
    return explanations

def assess_safety(text: str, selected: list[tuple[int, float]]):
    """Flag high-distress text so the UI can show extra care language."""
    text_lower = text.lower()
    matched_terms = sorted(term for term in DISTRESS_TERMS if term in text_lower)
    high_distress_emotions = {
        labels[index]
        for index, probability in selected
        if probability >= 0.45 and labels[index] in {"grief", "sadness", "fear", "remorse"}
    }
    show_support_message = bool(matched_terms or high_distress_emotions)
    return {
        "show_support_message": show_support_message,
        "matched_terms": matched_terms,
        "high_distress_emotions": sorted(high_distress_emotions),
    }

def calculate_arousal(probabilities):
    """Calculate emotional arousal (0 to 1)"""
    # Map arousal based on specific emotions
    high_arousal_emotions = ["excitement", "anger", "fear", "nervousness", "surprise"]
    low_arousal_emotions = ["relief", "sadness", "grief", "calm", "neutral"]
    
    high_arousal_indices = [labels.index(e) for e in high_arousal_emotions if e in labels]
    low_arousal_indices = [labels.index(e) for e in low_arousal_emotions if e in labels]
    
    high_arousal = sum(probabilities[i] for i in high_arousal_indices if i < len(probabilities))
    low_arousal = sum(probabilities[i] for i in low_arousal_indices if i < len(probabilities))
    
    total = high_arousal + low_arousal
    if total == 0:
        return 0.5  # Neutral arousal
    
    arousal = high_arousal / total
    
    # Adjust arousal based on overall emotion intensity
    max_prob = max(probabilities)
    if max_prob > 0.7:
        arousal = min(arousal * 1.3, 1.0)
    
    return arousal

def get_mock_emotions(text: str):
    """Fallback function that returns mock emotion data"""
    text_lower = text.lower()
    
    # Simple keyword-based emotion detection
    emotions = []
    if any(word in text_lower for word in ['happy', 'good', 'great', 'excited', 'joy']):
        emotions.append({"emotion": "joy", "probability": 0.6})
        emotions.append({"emotion": "excitement", "probability": 0.4})
        valence = 0.7
        arousal = 0.6
        label = "joy"
    elif any(word in text_lower for word in ['sad', 'upset', 'disappointed', 'hurt']):
        emotions.append({"emotion": "sadness", "probability": 0.6})
        emotions.append({"emotion": "disappointment", "probability": 0.4})
        valence = -0.5
        arousal = 0.3
        label = "sadness"
    elif any(word in text_lower for word in ['angry', 'mad', 'frustrated', 'annoyed']):
        emotions.append({"emotion": "anger", "probability": 0.6})
        emotions.append({"emotion": "annoyance", "probability": 0.4})
        valence = -0.6
        arousal = 0.8
        label = "anger"
    elif any(word in text_lower for word in ['love', 'care', 'appreciate', 'thankful']):
        emotions.append({"emotion": "love", "probability": 0.6})
        emotions.append({"emotion": "gratitude", "probability": 0.4})
        valence = 0.8
        arousal = 0.4
        label = "love"
    elif any(word in text_lower for word in ['scared', 'worried', 'anxious', 'nervous']):
        emotions.append({"emotion": "fear", "probability": 0.5})
        emotions.append({"emotion": "nervousness", "probability": 0.5})
        valence = -0.4
        arousal = 0.7
        label = "fear"
    else:
        emotions.append({"emotion": "neutral", "probability": 0.5})
        emotions.append({"emotion": "curiosity", "probability": 0.3})
        valence = 0.1
        arousal = 0.3
        label = "neutral"
    
    # Fill with some random emotions
    all_emotions = [e for e in labels if e not in [em["emotion"] for em in emotions]]
    import random
    for _ in range(4):
        emotion = random.choice(all_emotions)
        prob = random.uniform(0.05, 0.2)
        emotions.append({"emotion": emotion, "probability": prob})
    
    # Normalize probabilities
    total = sum(e["probability"] for e in emotions)
    for e in emotions:
        e["probability"] = e["probability"] / total
    
    emotions.sort(key=lambda x: x["probability"], reverse=True)
    
    return {
        "label": label,
        "probability": emotions[0]["probability"],
        "emotions": emotions[:6],
        "explanations": explain_prediction(text, [(labels.index(e["emotion"]), e["probability"]) for e in emotions[:4] if e["emotion"] in labels]),
        "safety": assess_safety(text, [(labels.index(e["emotion"]), e["probability"]) for e in emotions[:4] if e["emotion"] in labels]),
        "valence": valence,
        "arousal": arousal
    }
