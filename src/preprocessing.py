"""Preprocessing module for social media posts.

Supports English and Hinglish (code-mixed), handles URLs, mentions,
hashtags, repeated characters, emojis, and negations without crashing on
None, empty strings, malformed input, or emoji-only posts.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Set, Tuple

# Common emoji mappings to sentiment and emotion signals
EMOJI_LEXICON: Dict[str, Dict[str, str]] = {
    # Happiness / Positive
    "😊": {"sentiment": "Positive", "emotion": "Happiness", "token": "happy_face"},
    "😃": {"sentiment": "Positive", "emotion": "Happiness", "token": "smiling_face"},
    "😄": {"sentiment": "Positive", "emotion": "Happiness", "token": "laughing_face"},
    "😁": {"sentiment": "Positive", "emotion": "Happiness", "token": "beaming_face"},
    "🥰": {"sentiment": "Positive", "emotion": "Happiness", "token": "love_face"},
    "😍": {"sentiment": "Positive", "emotion": "Happiness", "token": "heart_eyes"},
    "❤️": {"sentiment": "Positive", "emotion": "Happiness", "token": "red_heart"},
    "💖": {"sentiment": "Positive", "emotion": "Happiness", "token": "sparkling_heart"},
    "🎉": {"sentiment": "Positive", "emotion": "Happiness", "token": "party_popper"},
    "✨": {"sentiment": "Positive", "emotion": "Happiness", "token": "sparkles"},
    "🚀": {"sentiment": "Positive", "emotion": "Happiness", "token": "rocket"},
    "👏": {"sentiment": "Positive", "emotion": "Happiness", "token": "clapping"},
    "🙌": {"sentiment": "Positive", "emotion": "Happiness", "token": "raising_hands"},
    "👍": {"sentiment": "Positive", "emotion": "Happiness", "token": "thumbs_up"},
    "🔥": {"sentiment": "Positive", "emotion": "Happiness", "token": "fire_lit"},
    "🏆": {"sentiment": "Positive", "emotion": "Happiness", "token": "trophy"},
    # Anger / Negative
    "😡": {"sentiment": "Negative", "emotion": "Anger", "token": "angry_face"},
    "🤬": {"sentiment": "Negative", "emotion": "Anger", "token": "cursing_face"},
    "👿": {"sentiment": "Negative", "emotion": "Anger", "token": "angry_devil"},
    "💢": {"sentiment": "Negative", "emotion": "Anger", "token": "anger_symbol"},
    # Sadness / Negative
    "😢": {"sentiment": "Negative", "emotion": "Sadness", "token": "crying_face"},
    "😭": {"sentiment": "Negative", "emotion": "Sadness", "token": "loudly_crying"},
    "💔": {"sentiment": "Negative", "emotion": "Sadness", "token": "broken_heart"},
    "😞": {"sentiment": "Negative", "emotion": "Sadness", "token": "disappointed"},
    "😔": {"sentiment": "Negative", "emotion": "Sadness", "token": "pensive"},
    "🥺": {"sentiment": "Negative", "emotion": "Sadness", "token": "pleading"},
    # Frustration / Negative
    "😤": {"sentiment": "Negative", "emotion": "Frustration", "token": "frustrated_steam"},
    "🤦": {"sentiment": "Negative", "emotion": "Frustration", "token": "facepalm"},
    "🤦‍♂️": {"sentiment": "Negative", "emotion": "Frustration", "token": "man_facepalm"},
    "🤦‍♀️": {"sentiment": "Negative", "emotion": "Frustration", "token": "woman_facepalm"},
    "🙄": {"sentiment": "Negative", "emotion": "Frustration", "token": "eye_roll"},
    "😩": {"sentiment": "Negative", "emotion": "Frustration", "token": "weary"},
    "😫": {"sentiment": "Negative", "emotion": "Frustration", "token": "tired_face"},
    # Fear / Negative
    "😨": {"sentiment": "Negative", "emotion": "Fear", "token": "scared_face"},
    "😱": {"sentiment": "Negative", "emotion": "Fear", "token": "screaming_fear"},
    "😰": {"sentiment": "Negative", "emotion": "Fear", "token": "anxious_face"},
    "😬": {"sentiment": "Negative", "emotion": "Fear", "token": "grimacing"},
    # Surprise / Positive or Neutral
    "😲": {"sentiment": "Neutral", "emotion": "Surprise", "token": "astonished_face"},
    "🤯": {"sentiment": "Neutral", "emotion": "Surprise", "token": "exploding_head"},
    "😮": {"sentiment": "Neutral", "emotion": "Surprise", "token": "open_mouth"},
    "😯": {"sentiment": "Neutral", "emotion": "Surprise", "token": "hushed_face"},
    "😳": {"sentiment": "Neutral", "emotion": "Surprise", "token": "flushed_face"},
    # Neutral / Respect
    "🙏": {"sentiment": "Positive", "emotion": "Happiness", "token": "folded_hands"},
    "☕": {"sentiment": "Positive", "emotion": "Happiness", "token": "coffee"},
}

# Negation words in English and Hinglish
NEGATION_WORDS: Set[str] = {
    "not", "never", "no", "neither", "nor", "none", "hardly", "scarcely",
    "barely", "isnt", "isn't", "arent", "aren't", "wasnt", "wasn't",
    "werent", "weren't", "havent", "haven't", "hasnt", "hasn't", "hadnt",
    "hadn't", "wont", "won't", "wouldnt", "wouldn't", "dont", "don't",
    "doesnt", "doesn't", "didnt", "didn't", "cant", "can't", "couldnt",
    "couldn't", "shouldnt", "shouldn't", "without",
    # Hinglish negations
    "nahi", "nahin", "nhi", "na", "mat", "ni", "kabhi nahi", "koi nahi",
    "bilkul nahi", "ghanta"
}

# Regex patterns compiled for efficiency
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@([a-zA-Z0-9_]+)")
HASHTAG_PATTERN = re.compile(r"#([a-zA-Z0-9_]+)")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{2,}")


def extract_emojis(text: str) -> List[Dict[str, str]]:
    """Extract known emojis with their mapped sentiment and emotion."""
    if not text or not isinstance(text, str):
        return []
    found = []
    for char in text:
        if char in EMOJI_LEXICON:
            found.append({
                "emoji": char,
                "sentiment": EMOJI_LEXICON[char]["sentiment"],
                "emotion": EMOJI_LEXICON[char]["emotion"],
                "token": EMOJI_LEXICON[char]["token"]
            })
    return found


def extract_mentions(text: str) -> List[str]:
    """Extract @mentions without the @ prefix."""
    if not text or not isinstance(text, str):
        return []
    return MENTION_PATTERN.findall(text)


def extract_hashtags(text: str) -> List[str]:
    """Extract hashtags without the # prefix."""
    if not text or not isinstance(text, str):
        return []
    return [h.lower() for h in HASHTAG_PATTERN.findall(text)]


def split_hashtag_words(hashtag: str) -> str:
    """Split PascalCase or camelCase hashtags into separate words.
    
    Example: 'BangaloreTraffic' -> 'bangalore traffic'
    """
    if not hashtag:
        return ""
    # Split camel/pascal casing: e.g. BangaloreTraffic -> Bangalore Traffic
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", hashtag)
    return s1.lower()


def normalize_repeated_characters(text: str) -> str:
    """Reduce 3 or more repeated characters to 2.
    
    Example: 'sooo goood' -> 'soo good', 'mastttt' -> 'mast' (if handled carefully).
    For characters repeated 3+ times, reduce to 2 characters to preserve common double-letters
    (like 'good', 'cool') while stripping exaggerations ('sooooo' -> 'soo', 'baddd' -> 'badd').
    """
    if not text:
        return ""
    # 3 or more consecutive identical characters -> 2
    return REPEATED_CHAR_PATTERN.sub(r"\1\1", text)


def clean_text(text: Any, preserve_case: bool = False) -> str:
    """Clean and normalize a social media post safely.
    
    Guarantees no crash on None, numbers, booleans, empty strings,
    malformed inputs, or emoji-only posts.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return ""

    # Normalize unicode characters (e.g. accents, full-width characters)
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs
    text = URL_PATTERN.sub(" ", text)

    # Normalize repeated characters (e.g. 'suuuuper' -> 'suuper')
    text = normalize_repeated_characters(text)

    # Replace @mentions with mention tokens or remove symbol
    text = MENTION_PATTERN.sub(r" \1 ", text)

    # For hashtags: preserve the text tokens inside the hashtag
    def replace_hashtag(match: re.Match) -> str:
        tag = match.group(1)
        return " " + split_hashtag_words(tag) + " "

    text = HASHTAG_PATTERN.sub(replace_hashtag, text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if not preserve_case:
        text = text.lower()

    return text


def tokenize(text: str) -> List[str]:
    """Tokenize cleaned text into word tokens, keeping hyphenated words and negations."""
    if not text:
        return []
    # Split on whitespace and non-alphanumeric except apostrophes
    tokens = re.findall(r"\b[a-zA-Z0-9_'\-]+\b", text)
    return [t.lower() for t in tokens if t]


def preprocess_post(raw_text: Any) -> Dict[str, Any]:
    """Complete preprocessing of a single social media post.
    
    Returns structured details including original text, cleaned text,
    extracted hashtags, mentions, emojis, and tokens.
    """
    text_str = "" if raw_text is None else str(raw_text)
    
    emojis = extract_emojis(text_str)
    hashtags = extract_hashtags(text_str)
    mentions = extract_mentions(text_str)
    cleaned = clean_text(text_str)
    tokens = tokenize(cleaned)

    return {
        "raw_text": text_str,
        "cleaned_text": cleaned,
        "hashtags": hashtags,
        "mentions": mentions,
        "emojis": emojis,
        "tokens": tokens,
        "has_negation": any(tok in NEGATION_WORDS for tok in tokens)
    }
