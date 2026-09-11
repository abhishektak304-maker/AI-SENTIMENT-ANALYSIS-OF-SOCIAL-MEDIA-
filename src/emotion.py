"""Emotion Classification Engine.

Supports 7 target emotions:
- Happiness
- Anger
- Sadness
- Frustration
- Fear
- Surprise
- Neutral

Provides a transparent deterministic multi-class scoring system with
English & Hinglish vocabulary, emoji mapping, and negation suppression.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple
from src.preprocessing import clean_text, extract_emojis, tokenize, NEGATION_WORDS

# Emotion Lexicons (English + Hinglish)
EMOTION_KEYWORDS: Dict[str, Dict[str, float]] = {
    "Happiness": {
        # English
        "happy": 2.0, "happiness": 2.0, "love": 2.0, "loved": 1.8, "loving": 1.6,
        "amazing": 1.8, "awesome": 1.8, "best": 1.8, "wonderful": 1.8, "joy": 2.0,
        "celebrating": 2.0, "celebration": 2.0, "proud": 2.0, "congrats": 1.8,
        "congratulations": 2.0, "blessed": 1.8, "bliss": 2.0, "triumph": 1.8,
        "victory": 1.8, "win": 1.5, "winning": 1.5, "champion": 1.8, "great": 1.2,
        "sweet": 1.2, "fantastic": 1.8, "delicious": 1.6, "peaceful": 1.6,
        "therapeutic": 1.6, "spiritual": 1.4, "laughing": 1.8, "smile": 1.5,
        "smiling": 1.5, "cherish": 1.6, "convenient": 1.2, "energized": 1.5,
        # Hinglish
        "maza": 1.8, "zabardast": 2.0, "mast": 1.8, "badiya": 1.8, "badhiya": 1.8,
        "lajawab": 2.0, "sukoon": 2.0, "shandar": 2.0, "shaandar": 2.0,
        "jhakaas": 2.0, "pyaar": 1.8, "pyar": 1.8, "kamaal": 1.8, "kamal": 1.8,
        "gazab": 1.8, "khushi": 2.0
    },
    "Anger": {
        # English
        "angry": 2.2, "furious": 2.5, "rage": 2.5, "outrage": 2.2, "boycott": 2.0,
        "corrupt": 2.0, "fraud": 2.2, "scam": 2.2, "cheat": 2.2, "cheated": 2.2,
        "illegal": 1.8, "hate": 2.0, "hated": 1.8, "disgust": 2.0, "disgusting": 2.0,
        "patronizing": 1.5, "shameless": 2.0, "shameful": 2.0, "screw": 1.8,
        "incompetent": 1.8, "screaming": 1.5,
        # Hinglish
        "gussa": 2.2, "krodh": 2.2, "ghatiya": 2.2, "chutiya": 2.5, "bewaqoof": 2.0,
        "ganda": 1.8, "gandi": 1.8, "beizzati": 2.0, "loot": 2.0, "danga": 2.0,
        "chor": 2.0
    },
    "Sadness": {
        # English
        "sad": 2.2, "sadness": 2.2, "sorrow": 2.2, "heartbreak": 2.5,
        "heartbroken": 2.5, "crying": 2.2, "cry": 2.0, "tears": 2.0,
        "depressed": 2.2, "depression": 2.2, "miss": 1.6, "missed": 1.6,
        "inmemoriam": 2.2, "rip": 2.0, "passing": 1.8, "loss": 1.8, "lost": 1.5,
        "devastating": 2.2, "grief": 2.2, "alone": 1.4, "neglected": 1.6,
        "pain": 1.6, "mourn": 2.0,
        # Hinglish
        "dukh": 2.2, "dukhi": 2.2, "udaas": 2.2, "udas": 2.2, "rona": 2.0,
        "aansu": 2.0, "toot": 1.8, "dard": 1.8
    },
    "Frustration": {
        # English
        "frustrated": 2.2, "frustrating": 2.2, "frustration": 2.2, "traffic": 1.6,
        "jammed": 1.8, "gridlock": 2.0, "delay": 1.8, "delayed": 1.8, "stuck": 1.8,
        "cancel": 1.6, "canceled": 1.6, "cancellation": 1.8, "outage": 1.8,
        "slow": 1.4, "wait": 1.4, "waiting": 1.4, "line": 1.2, "queue": 1.2,
        "harassed": 2.0, "harassment": 2.0, "refund": 1.5, "complaint": 1.6,
        "ticket": 1.2, "unresolved": 1.8, "annoyed": 2.0, "annoying": 2.0,
        "bureaucracy": 1.8, "hassle": 1.8, "useless": 1.6, "pathetic": 1.8,
        "terrible": 1.6, "broken": 1.4, "pothole": 1.5, "potholes": 1.5,
        "worst": 1.5, "sucks": 1.8, "refused": 1.6, "robotic": 1.6,
        # Hinglish
        "pareshan": 2.0, "pareshaan": 2.0, "bakwas": 1.8, "bakwaas": 1.8,
        "bekaar": 1.8, "bekar": 1.8, "faltu": 1.8, "faaltu": 1.8, "tang": 1.8,
        "kharab": 1.6, "chakkar": 1.4, "jhanjhat": 1.8, "dimag kharab": 2.2
    },
    "Fear": {
        # English
        "fear": 2.2, "scared": 2.2, "terrified": 2.5, "terror": 2.2, "panic": 2.5,
        "danger": 2.0, "dangerous": 2.0, "hazardous": 2.0, "hazards": 2.0,
        "accident": 2.0, "crash": 2.0, "collision": 2.0, "storm": 1.8,
        "earthquake": 2.0, "tremors": 1.8, "emergency": 2.0, "suffocating": 2.2,
        "warning": 1.6, "alert": 1.6, "threat": 2.0, "recklessly": 1.8,
        "dark": 1.2, "trapped": 2.0, "burst": 1.6, "vandalized": 1.8,
        # Hinglish
        "dar": 2.2, "darr": 2.2, "khatra": 2.0, "bhay": 2.0, "khauf": 2.2
    },
    "Surprise": {
        # English
        "surprise": 2.0, "surprised": 2.0, "shocking": 2.2, "shocked": 2.2,
        "shock": 2.0, "disbelief": 2.0, "unexpected": 2.0, "twist": 1.8,
        "unbelievable": 2.0, "mindblowing": 2.2, "cliffhanger": 1.8, "wow": 1.8,
        "whoa": 1.8, "miracle": 1.8, "stunned": 2.0, "astonished": 2.0,
        # Hinglish
        "hairan": 2.2, "heran": 2.2, "chamatkar": 2.0, "ajeeb": 1.4
    }
}


def analyze_emotion(raw_text: Any) -> Dict[str, Any]:
    """Classify text into one of seven supported emotions.
    
    Returns:
        {
            "emotion": "Happiness" | "Anger" | "Sadness" | "Frustration" | "Fear" | "Surprise" | "Neutral",
            "confidence": float (0.50 to 0.99),
            "scores": Dict[str, float]
        }
    """
    if raw_text is None or not str(raw_text).strip():
        return {
            "emotion": "Neutral",
            "confidence": 0.50,
            "scores": {e: 0.0 for e in ["Happiness", "Anger", "Sadness", "Frustration", "Fear", "Surprise", "Neutral"]}
        }

    text_str = str(raw_text)
    cleaned = clean_text(text_str)
    tokens = tokenize(cleaned)
    emojis = extract_emojis(text_str)

    scores: Dict[str, float] = {
        "Happiness": 0.0,
        "Anger": 0.0,
        "Sadness": 0.0,
        "Frustration": 0.0,
        "Fear": 0.0,
        "Surprise": 0.0,
        "Neutral": 0.2  # slight neutral baseline
    }

    # 1. Emoji signals
    for emo in emojis:
        emotion_name = emo.get("emotion")
        if emotion_name and emotion_name in scores:
            scores[emotion_name] += 1.8

    # 2. Token evaluation with negation suppression
    n_tokens = len(tokens)
    for i, token in enumerate(tokens):
        # Check if preceding or succeeding token is a negation (Hinglish SOV syntax)
        is_negated = False
        for step in (-1, -2, 1, 2):
            target_idx = i + step
            if 0 <= target_idx < n_tokens and tokens[target_idx] in NEGATION_WORDS:
                is_negated = True
                break

        for emotion_name, lex in EMOTION_KEYWORDS.items():
            if token in lex:
                val = lex[token]
                if is_negated:
                    # Negating an emotion suppresses that emotion or adds slight neutral/opposite
                    scores[emotion_name] -= val * 0.5
                else:
                    scores[emotion_name] += val

    # Handle composite expressions like "dimag kharab"
    joined_text = " ".join(tokens)
    if "dimag kharab" in joined_text:
        scores["Frustration"] += 2.5
    if "maza aa gaya" in joined_text or "bohot badiya" in joined_text:
        scores["Happiness"] += 2.5
    if "miss you" in joined_text:
        scores["Sadness"] += 2.0

    # Determine highest scoring emotion
    top_emotion, top_score = max(scores.items(), key=lambda item: item[1])

    # If top score is below threshold or tied near zero, default to Neutral
    if top_score <= 0.4:
        top_emotion = "Neutral"
        confidence = 0.65
    else:
        # Calculate confidence from top score relative to total non-negative scores
        total_positive = sum(max(0.0, s) for s in scores.values())
        if total_positive > 0:
            ratio = top_score / total_positive
            confidence = min(0.96, max(0.55, 0.50 + ratio * 0.45))
        else:
            confidence = 0.60

    return {
        "emotion": top_emotion,
        "confidence": round(confidence, 2),
        "scores": {k: round(v, 2) for k, v in scores.items()}
    }
