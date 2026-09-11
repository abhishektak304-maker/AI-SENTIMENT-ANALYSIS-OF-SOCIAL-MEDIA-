"""Sentiment Analysis Engine.

Provides transparent rule-based and lexical sentiment scoring with extensive
support for English and Hinglish (code-mixed), contextual negation flipping,
intensity scaling, and emoji integration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from src.preprocessing import clean_text, extract_emojis, tokenize, NEGATION_WORDS

# Positive lexical weights (English + Hinglish)
POSITIVE_WORDS: Dict[str, float] = {
    # English
    "amazing": 2.0, "awesome": 2.0, "great": 1.5, "love": 2.0, "loved": 1.8,
    "loving": 1.5, "lovely": 1.5, "best": 2.0, "good": 1.0, "fantastic": 2.0,
    "superb": 2.0, "brilliant": 2.0, "excellent": 2.0, "incredible": 2.0,
    "wonderful": 1.8, "happy": 1.5, "happiness": 1.5, "delight": 1.5,
    "delighted": 1.6, "bliss": 1.8, "enjoy": 1.4, "enjoyed": 1.4,
    "win": 1.5, "winner": 1.5, "winning": 1.5, "champion": 1.8, "top": 1.0,
    "perfect": 2.0, "masterpiece": 2.2, "promising": 1.2, "smooth": 1.2,
    "inspiring": 1.5, "proud": 1.6, "congrats": 1.5, "congratulations": 1.6,
    "respect": 1.4, "sweet": 1.0, "fast": 0.8, "convenient": 1.2,
    "helpful": 1.2, "mindblowing": 2.2, "delicious": 1.8, "tasty": 1.4,
    "peaceful": 1.5, "therapeutic": 1.5, "beautiful": 1.6, "clean": 1.0,
    "sparkling": 1.2, "kindness": 1.5, "miracle": 1.8, "dedication": 1.4,
    "distinction": 1.4, "success": 1.8, "nuanced": 1.0, "fresh": 1.0,
    "solid": 1.2, "worth": 1.2, "value": 1.0, "pleased": 1.4,
    # Hinglish
    "mast": 2.0, "zabardast": 2.2, "accha": 1.4, "achha": 1.4, "achi": 1.4,
    "achhi": 1.4, "badiya": 1.8, "badhiya": 1.8, "shandar": 2.0, "shaandar": 2.0,
    "lajawab": 2.0, "kamaal": 1.8, "kamal": 1.8, "jhakaas": 2.0, "gazab": 1.8,
    "fadu": 2.0, "faadu": 2.0, "maza": 1.6, "mazedar": 1.8, "changa": 1.4,
    "dhamaka": 1.5, "sahi": 1.0, "pyar": 1.6, "pyaar": 1.6, "sukoon": 1.8,
    "kadak": 1.5, "dhansu": 1.8, "bawaal": 1.5
}

# Negative lexical weights (English + Hinglish)
NEGATIVE_WORDS: Dict[str, float] = {
    # English
    "worst": -2.2, "hate": -2.0, "hated": -1.8, "terrible": -2.0, "angry": -1.8,
    "bad": -1.2, "awful": -2.0, "horrible": -2.0, "pathetic": -2.0,
    "disaster": -2.0, "poor": -1.2, "trash": -1.8, "garbage": -1.8,
    "useless": -1.8, "broken": -1.5, "scam": -2.2, "fraud": -2.2,
    "fail": -1.5, "failed": -1.5, "failure": -1.6, "cheat": -2.0,
    "cheated": -2.0, "disappointed": -1.6, "disappointing": -1.6, "sad": -1.5,
    "sadness": -1.5, "crying": -1.4, "pain": -1.4, "hurt": -1.4,
    "hurts": -1.4, "annoying": -1.5, "annoyed": -1.5, "frustrated": -1.8,
    "frustrating": -1.8, "frustration": -1.8, "nightmare": -2.0, "sucks": -1.8,
    "suffocating": -1.8, "panic": -1.8, "fear": -1.5, "scared": -1.6,
    "terrified": -1.8, "harassment": -2.0, "pothole": -1.2, "potholes": -1.2,
    "cancel": -1.2, "canceled": -1.2, "cancellation": -1.2, "delay": -1.2,
    "delayed": -1.2, "gridlock": -1.5, "jammed": -1.4, "shameful": -1.8,
    "vandalized": -1.8, "stale": -1.4, "soggy": -1.2, "defective": -1.6,
    "brunt": -1.2, "cracked": -1.4, "bricked": -2.0, "corrupt": -2.0,
    "unacceptable": -1.8, "recklessly": -1.8, "terror": -2.0, "deflated": -1.2,
    "neglected": -1.5, "ruined": -1.8, "choking": -1.8,
    # Hinglish
    "bakwas": -2.0, "bakwaas": -2.0, "bekaar": -1.8, "bekar": -1.8,
    "kharab": -1.8, "khrab": -1.8, "ghatiya": -2.2, "faltu": -1.6,
    "faaltu": -1.6, "pareshan": -1.6, "pareshaan": -1.6, "chutiya": -2.2,
    "bewaqoof": -1.8, "ganda": -1.6, "gandi": -1.6, "beizzati": -1.8,
    "raddi": -1.6, "loot": -1.8, "barbad": -2.0, "barbaad": -2.0,
    "jhuth": -1.8, "musibat": -1.6, "tang": -1.4, "chot": -1.4,
    "dhokha": -2.0
}

# Modifiers / Intensifiers
INTENSIFIERS: Dict[str, float] = {
    "very": 1.5, "extremely": 1.8, "really": 1.4, "super": 1.5,
    "totally": 1.5, "absolutely": 1.7, "hugely": 1.5, "so": 1.3,
    "too": 1.3, "unbelievably": 1.8, "completely": 1.5,
    # Hinglish intensifiers
    "bohot": 1.6, "bahut": 1.6, "bht": 1.5, "bilkul": 1.5,
    "ekdam": 1.6, "pura": 1.4, "poora": 1.4, "sabse": 1.5,
    "full": 1.3
}


def analyze_sentiment(raw_text: Any) -> Dict[str, Any]:
    """Analyze sentiment of a post with transparent lexical rules.
    
    Returns:
        {
            "label": "Positive" | "Negative" | "Neutral",
            "confidence": float (0.50 to 0.99),
            "score": float (-1.0 to 1.0),
            "method": "lexical_hinglish_negation_v1"
        }
    """
    if raw_text is None or not str(raw_text).strip():
        return {
            "label": "Neutral",
            "confidence": 0.50,
            "score": 0.0,
            "method": "fallback_empty"
        }

    text_str = str(raw_text)
    cleaned = clean_text(text_str)
    tokens = tokenize(cleaned)
    emojis = extract_emojis(text_str)

    if not tokens and not emojis:
        return {
            "label": "Neutral",
            "confidence": 0.50,
            "score": 0.0,
            "method": "fallback_no_tokens"
        }

    total_score = 0.0
    matched_signals = 0

    # 1. Emoji Contribution
    for emo in emojis:
        if emo["sentiment"] == "Positive":
            total_score += 1.5
            matched_signals += 1
        elif emo["sentiment"] == "Negative":
            total_score -= 1.5
            matched_signals += 1

    # 2. Token Scoring with Negation Window & Intensifiers
    n_tokens = len(tokens)
    for i, token in enumerate(tokens):
        base_val = 0.0
        is_pos = token in POSITIVE_WORDS
        is_neg = token in NEGATIVE_WORDS

        if is_pos:
            base_val = POSITIVE_WORDS[token]
        elif is_neg:
            base_val = NEGATIVE_WORDS[token]
        else:
            continue

        # Check for intensifier in previous 2 tokens
        multiplier = 1.0
        for back_step in (1, 2):
            if i - back_step >= 0:
                prev_tok = tokens[i - back_step]
                if prev_tok in INTENSIFIERS:
                    multiplier *= INTENSIFIERS[prev_tok]

        # Check for negation in previous 1 to 3 tokens OR next 1 to 2 tokens (Hinglish SOV syntax)
        is_negated = False
        for step in (-1, -2, -3, 1, 2):
            target_idx = i + step
            if 0 <= target_idx < n_tokens:
                neighbor_tok = tokens[target_idx]
                if neighbor_tok in NEGATION_WORDS:
                    is_negated = True
                    break

        if is_negated:
            # Flips sentiment:
            # "not good" -> negative (-0.8 * base_val)
            # "not bad" -> positive (-0.7 * base_val)
            # "bilkul bekar nahi" -> positive/neutral (-0.7 * base_val)
            effective_val = -0.75 * base_val * multiplier
        else:
            effective_val = base_val * multiplier

        total_score += effective_val
        matched_signals += 1

    # 3. Label and Confidence Calibration
    # Normalize score approximately between -1.0 and 1.0
    if matched_signals > 0:
        normalized_score = max(-1.0, min(1.0, total_score / (matched_signals * 1.5)))
    else:
        normalized_score = 0.0

    threshold = 0.15
    if normalized_score > threshold:
        label = "Positive"
        # Confidence increases with score magnitude and signal density
        confidence = min(0.98, 0.65 + abs(normalized_score) * 0.32)
    elif normalized_score < -threshold:
        label = "Negative"
        confidence = min(0.98, 0.65 + abs(normalized_score) * 0.32)
    else:
        label = "Neutral"
        # High confidence for neutral if no affective signals detected
        confidence = 0.75 if matched_signals == 0 else 0.60

    return {
        "label": label,
        "confidence": round(confidence, 2),
        "score": round(normalized_score, 3),
        "signals_count": matched_signals,
        "method": "transparent_lexical_hinglish_v1"
    }
