"""SQLite Database Management Module for SocialPulse.

Provides thread-safe initialization, schema management, parameterized querying,
and batch insertion with strict SQL injection prevention, explicit connection
cleanup, and error handling.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple
import pandas as pd

DEFAULT_DB_PATH = os.path.join("data", "socialpulse.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    post_id INTEGER,
    timestamp TEXT NOT NULL,
    author TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    cleaned_text TEXT,
    source TEXT DEFAULT 'manual',
    sentiment TEXT NOT NULL,
    sentiment_score REAL DEFAULT 0.0,
    sentiment_confidence REAL DEFAULT 0.5,
    emotion TEXT NOT NULL,
    emotion_confidence REAL DEFAULT 0.5,
    topic TEXT NOT NULL,
    topic_confidence REAL DEFAULT 0.5,
    hashtags TEXT DEFAULT '',
    mentions TEXT DEFAULT '',
    keywords TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_posts_timestamp ON posts(timestamp);
CREATE INDEX IF NOT EXISTS idx_posts_sentiment ON posts(sentiment);
CREATE INDEX IF NOT EXISTS idx_posts_emotion ON posts(emotion);
CREATE INDEX IF NOT EXISTS idx_posts_topic ON posts(topic);

CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


def get_db_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Create and return a configured sqlite3 connection."""
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


@contextmanager
def open_db(db_path: str = DEFAULT_DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Context manager ensuring sqlite3 connection is committed and explicitly closed."""
    conn = get_db_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str = DEFAULT_DB_PATH, seed_from_csv: bool = True) -> Tuple[bool, str]:
    """Check if database exists; if not or if empty, create schema and seed it.
    
    Returns:
        (created_or_initialized: bool, message: str)
    """
    db_existed = os.path.exists(db_path)
    try:
        with open_db(db_path) as conn:
            conn.executescript(SCHEMA_SQL)
            
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM posts;")
            count = cur.fetchone()[0]

            if count == 0 and seed_from_csv:
                from src.analysis import analyze_dataframe
                sample_csv_path = os.path.join("data", "sample_posts.csv")
                if not os.path.exists(sample_csv_path):
                    sample_csv_path = "sample_posts.csv"

                if os.path.exists(sample_csv_path):
                    raw_df = pd.read_csv(sample_csv_path)
                    analyzed_df = analyze_dataframe(raw_df)
                    insert_posts_batch(analyzed_df, db_path=db_path)
                    msg = f"Database created and seeded with {len(analyzed_df)} records from sample data."
                    _log_event(conn, "INIT", msg)
                    conn.commit()
                    return True, msg
                else:
                    msg = "Database schema initialized (empty; sample CSV not found)."
                    _log_event(conn, "INIT", msg)
                    conn.commit()
                    return True, msg
            
            conn.commit()
            status = "Database verified and ready." if db_existed else "Database created successfully."
            return True, status

    except Exception as e:
        return False, f"Failed to initialize database safely: {type(e).__name__}"


def _log_event(conn: sqlite3.Connection, event_type: str, message: str) -> None:
    """Internal helper to record system audit logs."""
    try:
        conn.execute(
            "INSERT INTO system_logs (event_type, message, timestamp) VALUES (?, ?, ?);",
            (event_type, message, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
        )
    except Exception:
        pass


def insert_post(post_data: Dict[str, Any], db_path: str = DEFAULT_DB_PATH) -> Optional[int]:
    """Insert a single analyzed post record using parameterized query."""
    hashtags = post_data.get("hashtags", [])
    if isinstance(hashtags, list):
        hashtags_val = " ".join([f"#{h}" if not h.startswith("#") else h for h in hashtags])
    else:
        hashtags_val = str(hashtags)

    mentions = post_data.get("mentions", [])
    if isinstance(mentions, list):
        mentions_val = " ".join([f"@{m}" if not m.startswith("@") else m for m in mentions])
    else:
        mentions_val = str(mentions)

    keywords = post_data.get("keywords", [])
    if isinstance(keywords, list):
        keywords_val = ", ".join(keywords)
    else:
        keywords_val = str(keywords)

    insert_sql = """
    INSERT INTO posts (
        post_id, timestamp, author, raw_text, cleaned_text, source,
        sentiment, sentiment_score, sentiment_confidence,
        emotion, emotion_confidence,
        topic, topic_confidence,
        hashtags, mentions, keywords, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    
    params = (
        int(post_data.get("post_id", 0)),
        str(post_data.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))),
        str(post_data.get("author", "anonymous")),
        str(post_data.get("raw_text", "")),
        str(post_data.get("cleaned_text", "")),
        str(post_data.get("source", "manual")),
        str(post_data.get("sentiment", "Neutral")),
        float(post_data.get("sentiment_score", 0.0)),
        float(post_data.get("sentiment_confidence", 0.5)),
        str(post_data.get("emotion", "Neutral")),
        float(post_data.get("emotion_confidence", 0.5)),
        str(post_data.get("topic", "Other")),
        float(post_data.get("topic_confidence", 0.5)),
        hashtags_val,
        mentions_val,
        keywords_val,
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    )

    try:
        with open_db(db_path) as conn:
            cur = conn.cursor()
            cur.execute(insert_sql, params)
            conn.commit()
            return cur.lastrowid
    except Exception:
        return None


def insert_posts_batch(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH) -> int:
    """Batch-insert multiple analyzed posts using parameterized transaction."""
    if df is None or df.empty:
        return 0

    insert_sql = """
    INSERT INTO posts (
        post_id, timestamp, author, raw_text, cleaned_text, source,
        sentiment, sentiment_score, sentiment_confidence,
        emotion, emotion_confidence,
        topic, topic_confidence,
        hashtags, mentions, keywords, created_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    records = []
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    for idx, row in df.iterrows():
        p_id = row.get("post_id", idx + 1)
        ts = row.get("timestamp", now_utc)
        auth = row.get("author", f"user_{idx+1}")
        raw_t = row.get("raw_text", "")
        clean_t = row.get("cleaned_text", "")
        src = row.get("source", "batch")
        sent = row.get("sentiment", "Neutral")
        sent_s = float(row.get("sentiment_score", 0.0))
        sent_c = float(row.get("sentiment_confidence", 0.5))
        emo = row.get("emotion", "Neutral")
        emo_c = float(row.get("emotion_confidence", 0.5))
        top = row.get("topic", "Other")
        top_c = float(row.get("topic_confidence", 0.5))

        ht = row.get("hashtags_str", "") or row.get("hashtags", "")
        if isinstance(ht, list):
            ht = " ".join([f"#{h}" for h in ht])
            
        mn = row.get("mentions", "")
        if isinstance(mn, list):
            mn = " ".join([f"@{m}" for m in mn])

        kw = row.get("keywords_str", "") or row.get("keywords", "")
        if isinstance(kw, list):
            kw = ", ".join(kw)

        records.append((
            int(p_id) if pd.notna(p_id) else (idx + 1),
            str(ts),
            str(auth),
            str(raw_t),
            str(clean_t),
            str(src),
            str(sent),
            sent_s,
            sent_c,
            str(emo),
            emo_c,
            str(top),
            top_c,
            str(ht),
            str(mn),
            str(kw),
            now_utc
        ))

    try:
        with open_db(db_path) as conn:
            cur = conn.cursor()
            cur.executemany(insert_sql, records)
            _log_event(conn, "BATCH_INSERT", f"Inserted {len(records)} records.")
            conn.commit()
            return len(records)
    except Exception:
        return 0


def get_all_posts(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Retrieve all posts from SQLite database formatted as DataFrame."""
    if not os.path.exists(db_path):
        init_db(db_path=db_path)

    query = """
    SELECT 
        post_id, timestamp, author, raw_text, cleaned_text, source,
        sentiment, sentiment_score, sentiment_confidence,
        emotion, emotion_confidence,
        topic, topic_confidence,
        hashtags, mentions, keywords
    FROM posts
    ORDER BY id ASC;
    """

    try:
        with open_db(db_path) as conn:
            df = pd.read_sql_query(query, conn)
            
            df["hashtags_str"] = df["hashtags"]
            df["hashtags"] = df["hashtags"].apply(
                lambda x: [tag.strip("#") for tag in str(x).split() if tag.startswith("#")] if pd.notna(x) else []
            )
            df["mentions"] = df["mentions"].apply(
                lambda x: [m.strip("@") for m in str(x).split() if m.startswith("@")] if pd.notna(x) else []
            )
            df["keywords_str"] = df["keywords"]
            df["keywords"] = df["keywords"].apply(
                lambda x: [k.strip() for k in str(x).split(",") if k.strip()] if pd.notna(x) else []
            )
            df["parsed_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            return df
    except Exception:
        return pd.DataFrame(columns=[
            "post_id", "timestamp", "author", "raw_text", "cleaned_text", "source",
            "sentiment", "sentiment_score", "sentiment_confidence",
            "emotion", "emotion_confidence", "topic", "topic_confidence",
            "hashtags", "hashtags_str", "mentions", "keywords", "keywords_str"
        ])


def get_db_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Return database diagnostic statistics safely."""
    if not os.path.exists(db_path):
        return {
            "exists": False,
            "path": db_path,
            "size_bytes": 0,
            "size_kb": 0.0,
            "total_posts": 0,
            "status": "Not Found"
        }

    size_bytes = os.path.getsize(db_path)
    try:
        with open_db(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM posts;")
            total_posts = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM system_logs;")
            total_logs = cur.fetchone()[0]

            return {
                "exists": True,
                "path": db_path,
                "size_bytes": size_bytes,
                "size_kb": round(size_bytes / 1024, 1),
                "total_posts": total_posts,
                "total_logs": total_logs,
                "status": "Healthy (Connected)"
            }
    except Exception as e:
        return {
            "exists": True,
            "path": db_path,
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 1),
            "total_posts": 0,
            "status": f"Error: {type(e).__name__}"
        }


def reset_db(db_path: str = DEFAULT_DB_PATH) -> Tuple[bool, str]:
    """Clear all records and re-seed from sample dataset."""
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
        return init_db(db_path=db_path, seed_from_csv=True)
    except Exception as e:
        return False, f"Failed to reset database: {type(e).__name__}"
