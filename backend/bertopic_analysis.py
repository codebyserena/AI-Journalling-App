# backend/bertopic_analysis.py
from bertopic import BERTopic

topic_model = BERTopic()

def extract_topics(texts: list[str]):
    topics, _ = topic_model.fit_transform(texts)
    return {"topics": topics}
