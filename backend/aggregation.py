from .db import SessionLocal, Prediction, PredictionFeedback
from datetime import datetime, timedelta
from collections import Counter, defaultdict

EMOTION_FIELDS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]

GROUPED_EMOTIONS = {
    "joy": ["joy", "amusement", "excitement", "optimism", "admiration", "pride", "surprise"],
    "calm": ["relief", "neutral", "realization", "approval", "curiosity"],
    "sadness": ["sadness", "grief", "disappointment", "remorse"],
    "anxiety": ["nervousness", "fear", "confusion"],
    "anger": ["anger", "annoyance", "disgust", "disapproval"],
    "love": ["love", "caring", "gratitude", "desire"],
}

POSITIVE_GROUPS = {"joy", "calm", "love"}
NEGATIVE_GROUPS = {"sadness", "anxiety", "anger"}


def grouped_scores(entry):
    return {
        group: sum((getattr(entry, emotion) or 0.0) for emotion in emotions)
        for group, emotions in GROUPED_EMOTIONS.items()
    }


def primary_raw_emotion(entry):
    return max(
        EMOTION_FIELDS,
        key=lambda emotion: getattr(entry, emotion) or 0.0,
    )


def primary_group(entry):
    scores = grouped_scores(entry)
    return max(scores.items(), key=lambda item: item[1])[0]


def normalize_scores(scores):
    total = sum(scores.values())
    if total <= 0:
        return {key: 0.0 for key in scores}
    return {key: value / total for key, value in scores.items()}


def positive_negative_balance(group_scores):
    positive = sum(group_scores[group] for group in POSITIVE_GROUPS)
    negative = sum(group_scores[group] for group in NEGATIVE_GROUPS)
    total = positive + negative
    if total <= 0:
        return {"positive": 0.0, "negative": 0.0, "balance": 0.0}
    return {
        "positive": positive / total,
        "negative": negative / total,
        "balance": (positive - negative) / total,
    }

def get_daily_aggregation(user_id: str):
    db = SessionLocal()
    today = datetime.utcnow().date()
    start_of_day = datetime(today.year, today.month, today.day)
    
    # Get today's entries
    entries = db.query(Prediction).filter(
        Prediction.user_id == user_id,
        Prediction.timestamp >= start_of_day
    ).all()
    
    if not entries:
        return {
            "date": today.isoformat(),
            "total_predictions": 0,
            "top_emotions": []
        }
    
    # Calculate emotion frequencies
    emotion_counts = defaultdict(int)
    for entry in entries:
        max_emotion = None
        max_prob = -1
        
        for emotion in EMOTION_FIELDS:
            prob = getattr(entry, emotion)
            if prob is not None and prob > max_prob:
                max_prob = prob
                max_emotion = emotion
        
        if max_emotion:
            emotion_counts[max_emotion] += 1
    
    # Convert to list and sort
    top_emotions = sorted(
        [{"emotion": k, "count": v} for k, v in emotion_counts.items()],
        key=lambda x: x["count"],
        reverse=True
    )[:5]
    
    return {
        "date": today.isoformat(),
        "total_predictions": len(entries),
        "top_emotions": top_emotions
    }

def get_monthly_aggregation(user_id: str):
    db = SessionLocal()
    today = datetime.utcnow()
    month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Get this month's entries
    entries = db.query(Prediction).filter(
        Prediction.user_id == user_id,
        Prediction.timestamp >= month_start
    ).all()
    
    if not entries:
        return {
            "month": today.strftime("%Y-%m"),
            "total_predictions": 0,
            "trends": [],
            "topics": [],
            "emotion_frequency": [],
            "monthly_common_emotions": [],
            "positive_negative_trend": [],
            "personal_insights": {
                "most_frequent_emotion": None,
                "most_improved_emotion": None,
                "most_recurring_emotion": None,
                "distress_banner": None,
            },
            "correction_analytics": {
                "total_feedback": 0,
                "most_corrected": [],
            },
        }
    
    weekly_data = defaultdict(lambda: {group: 0.0 for group in GROUPED_EMOTIONS})
    weekly_counts = defaultdict(int)
    emotion_frequency = Counter()

    for entry in entries:
        week_num = entry.timestamp.isocalendar()[1]
        week_key = f"Week {week_num - month_start.isocalendar()[1] + 1}"

        scores = grouped_scores(entry)
        for group, score in scores.items():
            weekly_data[week_key][group] += score
        weekly_counts[week_key] += 1
        emotion_frequency[primary_group(entry)] += 1
    
    # Convert to list and normalize
    trends = []
    positive_negative_trend = []
    for week, data in sorted(weekly_data.items()):
        normalized = normalize_scores(data)
        normalized["week"] = week
        normalized["entryCount"] = weekly_counts[week]
        trends.append(normalized)

        balance = positive_negative_balance(data)
        balance["week"] = week
        positive_negative_trend.append(balance)
    
    # If no trends created (all zeros), create a default trend
    if not trends:
        # Create a default trend for the current week
        trends.append({
            "week": "Week 1",
            "joy": 0.25,
            "calm": 0.25,
            "sadness": 0.1,
            "anxiety": 0.1,
            "anger": 0.1,
            "love": 0.2
        })
    
    # Calculate topics
    topics = extract_topics(entries)
    monthly_common_emotions = [
        {"emotion": emotion, "count": count, "percentage": count / len(entries)}
        for emotion, count in emotion_frequency.most_common(6)
    ]
    insights = build_personal_insights(trends, emotion_frequency, entries)
    correction_analytics = get_correction_analytics(user_id, month_start)
    
    return {
        "month": today.strftime("%Y-%m"),
        "total_predictions": len(entries),
        "trends": trends,
        "topics": topics,
        "emotion_frequency": monthly_common_emotions,
        "monthly_common_emotions": monthly_common_emotions,
        "positive_negative_trend": positive_negative_trend,
        "personal_insights": insights,
        "correction_analytics": correction_analytics,
    }


def build_personal_insights(trends, emotion_frequency, entries):
    most_frequent = None
    if emotion_frequency:
        emotion, count = emotion_frequency.most_common(1)[0]
        most_frequent = {
            "emotion": emotion,
            "count": count,
            "percentage": count / len(entries),
        }

    most_recurring = most_frequent
    most_improved = None
    if len(trends) >= 2:
        first = trends[0]
        last = trends[-1]
        deltas = {
            emotion: (last.get(emotion, 0.0) - first.get(emotion, 0.0))
            for emotion in GROUPED_EMOTIONS
        }
        positive_improvements = {
            emotion: delta for emotion, delta in deltas.items()
            if emotion in POSITIVE_GROUPS or (emotion in NEGATIVE_GROUPS and delta < 0)
        }
        if positive_improvements:
            emotion, delta = max(positive_improvements.items(), key=lambda item: abs(item[1]))
            most_improved = {
                "emotion": emotion,
                "change": delta,
                "direction": "increased" if delta >= 0 else "decreased",
            }

    distress_entries = [
        entry for entry in entries[-10:]
        if primary_group(entry) in {"sadness", "anxiety"}
        or (entry.grief or 0.0) >= 0.35
        or (entry.fear or 0.0) >= 0.35
        or (entry.sadness or 0.0) >= 0.35
    ]
    distress_banner = None
    if len(distress_entries) >= 3:
        distress_banner = {
            "show": True,
            "message": "You have expressed sadness, fear, grief, or anxiety in several recent entries. This tool is for reflection only and is not medical advice.",
            "entry_count": len(distress_entries),
        }

    return {
        "most_frequent_emotion": most_frequent,
        "most_improved_emotion": most_improved,
        "most_recurring_emotion": most_recurring,
        "distress_banner": distress_banner,
    }


def get_correction_analytics(user_id: str, month_start: datetime):
    db = SessionLocal()
    feedback_rows = db.query(PredictionFeedback, Prediction).join(
        Prediction,
        Prediction.id == PredictionFeedback.prediction_id
    ).filter(
        PredictionFeedback.user_id == user_id,
        PredictionFeedback.created_at >= month_start,
    ).all()

    correction_counts = Counter()
    for feedback, prediction in feedback_rows:
        if feedback.rating != "wrong" or not feedback.corrected_emotion:
            continue
        predicted = primary_raw_emotion(prediction)
        correction_counts[(predicted, feedback.corrected_emotion)] += 1

    return {
        "total_feedback": len(feedback_rows),
        "most_corrected": [
            {
                "from": source,
                "to": target,
                "count": count,
            }
            for (source, target), count in correction_counts.most_common(5)
        ],
    }

def extract_topics(entries):
    """Extract topics from entries"""
    from collections import Counter
    import re
    
    if not entries:
        return []
    
    all_words = []
    for entry in entries[:20]:  # Limit for performance
        words = re.findall(r'\b[a-z]{4,}\b', entry.text.lower())
        stop_words = {'that', 'with', 'have', 'this', 'from', 'they', 'what', 
                     'when', 'were', 'their', 'will', 'would', 'there'}
        words = [w for w in words if w not in stop_words]
        all_words.extend(words)
    
    if not all_words:
        return []
    
    word_counts = Counter(all_words)
    top_words = word_counts.most_common(10)
    
    # Group into topics
    topics = []
    for i, (word, count) in enumerate(top_words[:4]):
        topics.append({
            "id": f"topic_{i}",
            "name": f"{word.capitalize()} & Reflection",
            "keywords": [word, "reflection", "thought"],
            "entryCount": min(count, len(entries)),
            "primaryEmotion": get_primary_emotion_for_word(word, entries)
        })
    
    return topics

def get_primary_emotion_for_word(word, entries):
    """Determine primary emotion associated with a word"""
    emotion_scores = {
        "joy": 0, "calm": 0, "sadness": 0, "anxiety": 0, "anger": 0, "love": 0
    }
    
    for entry in entries:
        if word in entry.text.lower():
            # Calm: relief + neutral + realization
            emotion_scores["calm"] += (entry.relief or 0) + (entry.neutral or 0) + (entry.realization or 0)
            # Joy: joy + amusement + excitement + optimism + admiration
            emotion_scores["joy"] += (entry.joy or 0) + (entry.amusement or 0) + (entry.excitement or 0) + (entry.optimism or 0) + (entry.admiration or 0)
            # Sadness: sadness + grief + disappointment + remorse
            emotion_scores["sadness"] += (entry.sadness or 0) + (entry.grief or 0) + (entry.disappointment or 0) + (entry.remorse or 0)
            # Anxiety: nervousness + fear + confusion
            emotion_scores["anxiety"] += (entry.nervousness or 0) + (entry.fear or 0) + (entry.confusion or 0)
            # Anger: anger + annoyance + disgust + disapproval
            emotion_scores["anger"] += (entry.anger or 0) + (entry.annoyance or 0) + (entry.disgust or 0) + (entry.disapproval or 0)
            # Love: love + caring + gratitude + desire
            emotion_scores["love"] += (entry.love or 0) + (entry.caring or 0) + (entry.gratitude or 0) + (entry.desire or 0)
    
    # Return the emotion with highest score, default to "calm"
    return max(emotion_scores.items(), key=lambda x: x[1])[0] or "calm"
