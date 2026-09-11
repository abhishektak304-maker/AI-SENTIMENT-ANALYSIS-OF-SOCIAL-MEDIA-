"""Comprehensive Analysis Pipeline for SocialPulse.

Handles:
- Single post analysis
- DataFrame batch processing
- Dynamic metric aggregations (sentiment %, emotion %, topic %, keywords)
- Sentiment over time time-series aggregations
- Sentiment spike detection
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.preprocessing import extract_hashtags, preprocess_post
from src.sentiment import analyze_sentiment
from src.emotion import analyze_emotion
from src.topics import classify_topic, extract_post_keywords, extract_corpus_keywords


def analyze_single_post(
    text: Any,
    timestamp: Optional[Any] = None,
    author: str = "anonymous",
    post_id: Optional[Any] = None,
    source: str = "manual"
) -> Dict[str, Any]:
    """Execute the full NLP pipeline on a single social media post.
    
    Safe on empty, null, or malformed inputs.
    """
    pre = preprocess_post(text)
    sent = analyze_sentiment(pre["raw_text"])
    emo = analyze_emotion(pre["raw_text"])
    top = classify_topic(pre["raw_text"])
    keywords = extract_post_keywords(pre["cleaned_text"], top_k=5)

    # Safe timestamp handling
    ts_str = ""
    if timestamp is not None and str(timestamp).strip():
        try:
            parsed_ts = pd.to_datetime(timestamp)
            ts_str = str(parsed_ts)
        except Exception:
            ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    else:
        ts_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "post_id": post_id if post_id is not None else 1,
        "timestamp": ts_str,
        "author": str(author) if author else "anonymous",
        "raw_text": pre["raw_text"],
        "cleaned_text": pre["cleaned_text"],
        "source": str(source) if source else "manual",
        "sentiment": sent["label"],
        "sentiment_score": sent["score"],
        "sentiment_confidence": sent["confidence"],
        "emotion": emo["emotion"],
        "emotion_confidence": emo["confidence"],
        "topic": top["topic"],
        "topic_confidence": top["confidence"],
        "hashtags": pre["hashtags"],
        "hashtags_str": " ".join([f"#{h}" for h in pre["hashtags"]]),
        "mentions": pre["mentions"],
        "keywords": keywords,
        "keywords_str": ", ".join(keywords)
    }


def analyze_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    timestamp_column: str = "timestamp",
    author_column: str = "author"
) -> pd.DataFrame:
    """Batch-analyze an entire pandas DataFrame of social media posts.
    
    Never raises an unhandled exception on missing columns or empty data.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "post_id", "timestamp", "author", "raw_text", "cleaned_text", "source",
            "sentiment", "sentiment_score", "sentiment_confidence",
            "emotion", "emotion_confidence", "topic", "topic_confidence",
            "hashtags", "hashtags_str", "mentions", "keywords", "keywords_str"
        ])

    working_df = df.copy()

    # Identify text column
    if text_column not in working_df.columns:
        # Fallback to first string/object column
        obj_cols = [c for c in working_df.columns if working_df[c].dtype == object]
        chosen_col = obj_cols[0] if obj_cols else working_df.columns[0]
    else:
        chosen_col = text_column

    # Normalize timestamps
    if timestamp_column in working_df.columns:
        working_df["parsed_timestamp"] = pd.to_datetime(working_df[timestamp_column], errors="coerce")
        # Fill missing timestamps sequentially
        base_time = pd.Timestamp.now()
        missing_mask = working_df["parsed_timestamp"].isna()
        if missing_mask.any():
            working_df.loc[missing_mask, "parsed_timestamp"] = [
                base_time - pd.Timedelta(hours=i) for i in range(missing_mask.sum())
            ]
    else:
        # Generate timestamps if missing
        base_time = pd.Timestamp.now()
        working_df["parsed_timestamp"] = [
            base_time - pd.Timedelta(hours=i) for i in range(len(working_df))
        ]

    # Process records
    analyzed_records = []
    for idx, row in working_df.iterrows():
        raw_val = row[chosen_col] if chosen_col in row else ""
        ts_val = row["parsed_timestamp"]
        auth_val = row[author_column] if author_column in row else f"user_{idx+1}"
        p_id = row["post_id"] if "post_id" in row else (idx + 1)
        src = row["source"] if "source" in row else "web"

        record = analyze_single_post(
            text=raw_val,
            timestamp=ts_val,
            author=auth_val,
            post_id=p_id,
            source=src
        )
        analyzed_records.append(record)

    result_df = pd.DataFrame(analyzed_records)
    result_df["parsed_timestamp"] = pd.to_datetime(result_df["timestamp"], errors="coerce")
    return result_df


def get_sentiment_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate dynamic sentiment metrics, counts, and percentages."""
    if df is None or df.empty:
        return {
            "total_posts": 0,
            "positive_count": 0,
            "positive_pct": 0.0,
            "negative_count": 0,
            "negative_pct": 0.0,
            "neutral_count": 0,
            "neutral_pct": 0.0,
            "net_sentiment": 0.0
        }

    total = len(df)
    counts = df["sentiment"].value_counts().to_dict()
    pos = counts.get("Positive", 0)
    neg = counts.get("Negative", 0)
    neu = counts.get("Neutral", 0)

    pos_pct = round((pos / total) * 100, 1)
    neg_pct = round((neg / total) * 100, 1)
    neu_pct = round((neu / total) * 100, 1)
    net_sentiment = round(((pos - neg) / total) * 100, 1)

    return {
        "total_posts": total,
        "positive_count": pos,
        "positive_pct": pos_pct,
        "negative_count": neg,
        "negative_pct": neg_pct,
        "neutral_count": neu,
        "neutral_pct": neu_pct,
        "net_sentiment": net_sentiment
    }


def get_emotion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute emotion frequency distribution and percentages."""
    all_emotions = ["Happiness", "Anger", "Sadness", "Frustration", "Fear", "Surprise", "Neutral"]
    if df is None or df.empty:
        return pd.DataFrame({
            "Emotion": all_emotions,
            "Count": [0] * len(all_emotions),
            "Percentage": [0.0] * len(all_emotions)
        })

    counts = df["emotion"].value_counts().to_dict()
    total = len(df)
    data = []
    for emo in all_emotions:
        cnt = counts.get(emo, 0)
        pct = round((cnt / total) * 100, 1) if total > 0 else 0.0
        data.append({"Emotion": emo, "Count": cnt, "Percentage": pct})

    return pd.DataFrame(data).sort_values(by="Count", ascending=False)


def get_topic_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute topic distribution and percentages."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Topic", "Count", "Percentage"])

    counts = df["topic"].value_counts().reset_index()
    counts.columns = ["Topic", "Count"]
    total = len(df)
    counts["Percentage"] = (counts["Count"] / total * 100).round(1)
    return counts


def get_keyword_trends(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Extract top keywords using TF-IDF across the current post set."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Keyword", "Score"])

    texts = df["raw_text"].dropna().tolist()
    kw_scores = extract_corpus_keywords(texts, top_n=top_n)
    if not kw_scores:
        return pd.DataFrame(columns=["Keyword", "Score"])

    res = pd.DataFrame(kw_scores, columns=["Keyword", "Score"])
    res["Score"] = res["Score"].round(2)
    return res


def get_hashtag_trends(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Extract top hashtags across the analyzed dataset."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Hashtag", "Count"])

    tags_list = []
    for tags in df["hashtags"]:
        if isinstance(tags, list):
            tags_list.extend(tags)

    if not tags_list:
        return pd.DataFrame(columns=["Hashtag", "Count"])

    tag_series = pd.Series(tags_list).value_counts().reset_index()
    tag_series.columns = ["Hashtag", "Count"]
    tag_series["Hashtag"] = tag_series["Hashtag"].apply(lambda x: f"#{x}")
    return tag_series.head(top_n)


def get_sentiment_over_time(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Aggregate sentiment counts over time (hourly or daily intervals)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Time", "Positive", "Negative", "Neutral", "Total"])

    work = df.copy()
    if "parsed_timestamp" not in work.columns:
        work["parsed_timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")

    work = work.dropna(subset=["parsed_timestamp"]).sort_values("parsed_timestamp")
    if work.empty:
        return pd.DataFrame(columns=["Time", "Positive", "Negative", "Neutral", "Total"])

    # Group by period
    work["period"] = work["parsed_timestamp"].dt.floor(freq)
    grouped = work.groupby(["period", "sentiment"]).size().unstack(fill_value=0)

    for col in ["Positive", "Negative", "Neutral"]:
        if col not in grouped.columns:
            grouped[col] = 0

    grouped["Total"] = grouped["Positive"] + grouped["Negative"] + grouped["Neutral"]
    grouped = grouped.reset_index().rename(columns={"period": "Time"})
    return grouped


def detect_sentiment_spikes(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify time periods with significant spikes in negative or positive sentiment."""
    time_df = get_sentiment_over_time(df, freq="D")
    if len(time_df) < 3:
        return []

    spikes = []
    # Calculate rolling or z-score for Negative posts
    neg_mean = time_df["Negative"].mean()
    neg_std = time_df["Negative"].std()

    if neg_std > 0:
        for _, row in time_df.iterrows():
            z = (row["Negative"] - neg_mean) / neg_std
            if z >= 1.5 and row["Negative"] >= 3:
                spikes.append({
                    "date": str(row["Time"].date()),
                    "type": "Negative Surge",
                    "negative_count": int(row["Negative"]),
                    "total_count": int(row["Total"]),
                    "message": f"High volume of negative sentiment detected ({int(row['Negative'])} posts)."
                })

    return spikes
