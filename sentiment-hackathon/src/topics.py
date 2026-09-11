"""Topic Classification and Keyword Extraction Module.

Maps social media posts into 11 categories:
- Technology
- Sports
- Food
- Travel
- Education
- Traffic
- Entertainment
- Shopping
- Customer Service
- Politics
- Other

Extracts hashtags and computes high-salience keywords using TF-IDF and frequency analysis.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import clean_text, extract_hashtags, tokenize

# Predefined domain vocabularies for 10 distinct categories + Other
TOPIC_LEXICONS: Dict[str, Set[str]] = {
    "Technology": {
        "ai", "tech", "technology", "software", "coding", "code", "python", "developer",
        "developers", "model", "weights", "screen", "laptop", "phone", "smartphone",
        "device", "gadget", "update", "cloud", "database", "servers", "algorithm",
        "keyboards", "keyboard", "prompt", "engineering", "llm", "fintech", "ux",
        "open-weights", "opensource", "open-source", "github", "devops", "outage",
        "quantum", "battery", "bug", "features", "hardware", "wifi", "broadband",
        "internet", "fiber"
    },
    "Sports": {
        "cricket", "ipl", "match", "team", "football", "goal", "goals", "derby",
        "tennis", "stadium", "tournament", "trophy", "champion", "champions",
        "skipper", "player", "players", "quarter", "finals", "stoppage", "score",
        "runs", "wicket", "wickets", "athlete", "athletic", "fitness", "cycling",
        "workout", "marathon"
    },
    "Food": {
        "biryani", "food", "foodie", "restaurant", "dining", "swiggy", "zomato",
        "momos", "chutney", "streetfood", "coffee", "chai", "tea", "cake", "bakery",
        "baking", "pizza", "curry", "lunch", "dinner", "thali", "sourdough", "bread",
        "dal", "makhani", "recipes", "snack", "paranthe", "jalebi", "rabri",
        "taste", "spicy", "delicious", "stale", "flavor"
    },
    "Travel": {
        "flight", "airline", "airport", "boarding", "luggage", "travel", "trip",
        "train", "railway", "railways", "station", "platform", "passengers",
        "hotel", "tourism", "vacay", "vacation", "hills", "mountains", "beach",
        "sea", "estate", "munnar", "gokarna", "rajasthan", "heritage", "journey",
        "trek", "hiking", "departure", "destination"
    },
    "Education": {
        "exam", "exams", "syllabus", "school", "college", "university", "student",
        "students", "professor", "degree", "thesis", "marksheet", "clerk", "study",
        "results", "curriculum", "scholarship", "campus", "academic", "distinction",
        "reform", "learning", "books", "biography", "workshop"
    },
    "Traffic": {
        "traffic", "jam", "jammed", "gridlock", "signal", "pothole", "potholes",
        "road", "roads", "highway", "junction", "commute", "metro", "cab", "uber",
        "auto", "driver", "ridehailing", "waterlogging", "bypass", "crossing",
        "flyover", "silk", "board", "outer", "ring", "transport"
    },
    "Entertainment": {
        "movie", "film", "cinema", "actor", "theatre", "acting", "music", "song",
        "remake", "concert", "guitar", "acoustics", "netflix", "documentary",
        "series", "show", "comedy", "satire", "streaming", "stand-up", "musical",
        "orchestra", "bollywood", "hollywood", "plot", "twist", "cliffhanger"
    },
    "Shopping": {
        "shopping", "order", "ordered", "package", "parcel", "delivery", "courier",
        "amazon", "flipkart", "sale", "discount", "discounts", "price", "refurbished",
        "defective", "return", "pickup", "seller", "cart", "product", "warranty",
        "groceries", "commerce", "quickcommerce", "unboxing"
    },
    "Customer Service": {
        "customer", "support", "ticket", "tickets", "complaint", "helpline",
        "executive", "agent", "refund", "refunded", "dispute", "unresolved",
        "grievance", "service", "care", "bot", "representative", "delay",
        "escalation", "call"
    },
    "Politics": {
        "politics", "government", "minister", "parliament", "election", "elections",
        "vote", "voter", "budget", "bill", "law", "policy", "scheme", "municipal",
        "corporation", "governance", "democracy", "constituency", "subsidy",
        "public", "civic", "pwd"
    }
}

# Stopwords to filter out during keyword extraction
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
    "by", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "of", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "can", "will", "just", "don", "should", "now", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "would", "could", "my", "your", "his",
    "her", "its", "our", "their", "me", "him", "them", "we", "us", "you", "i",
    "he", "she", "it", "they", "hai", "ho", "ka", "ki", "ke", "ko", "se", "me",
    "mein", "pe", "par", "bhi", "thi", "tha", "the", "kar", "karo", "kya", "yeh",
    "woh", "bilkul", "ekdam", "bohot", "bahut"
}


def classify_topic(raw_text: Any) -> Dict[str, Any]:
    """Classify a social media post into one of 11 predefined topics.
    
    Returns:
        {
            "topic": str,
            "confidence": float,
            "matched_terms": List[str]
        }
    """
    if raw_text is None or not str(raw_text).strip():
        return {
            "topic": "Other",
            "confidence": 0.50,
            "matched_terms": []
        }

    text_str = str(raw_text)
    cleaned = clean_text(text_str)
    tokens = tokenize(cleaned)
    hashtags = extract_hashtags(text_str)

    # Combine tokens and hashtag constituents
    all_tokens = set(tokens)
    for tag in hashtags:
        all_tokens.add(tag)
        for part in tag.split("_"):
            all_tokens.add(part)

    scores: Dict[str, float] = {t: 0.0 for t in TOPIC_LEXICONS.keys()}
    term_matches: Dict[str, List[str]] = {t: [] for t in TOPIC_LEXICONS.keys()}

    for topic, lexicon in TOPIC_LEXICONS.items():
        for token in all_tokens:
            if token in lexicon:
                scores[topic] += 1.5
                term_matches[topic].append(token)

    # Contextual multi-word checks
    text_lower = text_str.lower()
    if "customer support" in text_lower or "customer care" in text_lower:
        scores["Customer Service"] += 2.5
        term_matches["Customer Service"].append("customer_service")
    if "traffic jam" in text_lower or "silk board" in text_lower:
        scores["Traffic"] += 2.5
        term_matches["Traffic"].append("traffic_jam")
    if "open source" in text_lower or "machine learning" in text_lower:
        scores["Technology"] += 2.5
        term_matches["Technology"].append("open_source")

    best_topic, best_score = max(scores.items(), key=lambda item: item[1])

    if best_score < 1.0:
        return {
            "topic": "Other",
            "confidence": 0.50,
            "matched_terms": []
        }

    confidence = min(0.95, 0.60 + (best_score / 10.0) * 0.35)
    return {
        "topic": best_topic,
        "confidence": round(confidence, 2),
        "matched_terms": term_matches[best_topic]
    }


def extract_post_keywords(text: str, top_k: int = 5) -> List[str]:
    """Extract notable keyword tokens from a single post."""
    if not text:
        return []
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    
    # Filter out stopwords and short tokens
    salient = [t for t in tokens if len(t) > 2 and t not in STOPWORDS]
    
    # Keep unique in order of appearance
    seen = set()
    ordered = []
    for s in salient:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
            if len(ordered) >= top_k:
                break
    return ordered


def extract_corpus_keywords(texts: List[str], top_n: int = 20) -> List[Tuple[str, float]]:
    """Extract top keywords across a list of posts using TF-IDF scoring.
    
    Returns list of (keyword, score) tuples.
    """
    valid_texts = [clean_text(t) for t in texts if t and clean_text(t).strip()]
    if not valid_texts:
        return []

    try:
        tfidf = TfidfVectorizer(
            max_features=top_n,
            stop_words=list(STOPWORDS),
            token_pattern=r"\b[a-zA-Z]{3,}\b"
        )
        matrix = tfidf.fit_transform(valid_texts)
        scores = matrix.sum(axis=0).A1
        words = tfidf.get_feature_names_out()

        keyword_scores = [(words[i], float(scores[i])) for i in range(len(words))]
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        return keyword_scores[:top_n]
    except Exception:
        # Fallback if TF-IDF fails (e.g. on very small corpus)
        freq: Dict[str, int] = {}
        for t in valid_texts:
            for tok in tokenize(t):
                if len(tok) > 2 and tok not in STOPWORDS:
                    freq[tok] = freq.get(tok, 0) + 1
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [(k, float(v)) for k, v in sorted_freq[:top_n]]
