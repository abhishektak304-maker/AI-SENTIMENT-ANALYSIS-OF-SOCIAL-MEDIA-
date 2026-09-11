"""Automated Unit & Integration Test Suite for SocialPulse."""

from __future__ import annotations

import io
import unittest
import pandas as pd

from src.preprocessing import (
    clean_text,
    extract_emojis,
    extract_hashtags,
    extract_mentions,
    normalize_repeated_characters,
    preprocess_post,
    split_hashtag_words,
)
from src.sentiment import analyze_sentiment
from src.emotion import analyze_emotion
from src.topics import classify_topic, extract_corpus_keywords, extract_post_keywords
from src.analysis import (
    analyze_dataframe,
    analyze_single_post,
    detect_sentiment_spikes,
    get_emotion_summary,
    get_sentiment_over_time,
    get_sentiment_summary,
    get_topic_summary,
)
from src.utils import (
    ActionRateLimiter,
    sanitize_input_text,
    validate_uploaded_file,
)


class TestSocialPulsePipeline(unittest.TestCase):
    """Test suite covering preprocessing, NLP engines, analysis, and safety."""

    def test_normal_english_positive(self):
        text = "This new product is absolutely amazing and fantastic!"
        res = analyze_single_post(text)
        self.assertEqual(res["sentiment"], "Positive")
        self.assertGreater(res["sentiment_score"], 0.2)
        self.assertEqual(res["emotion"], "Happiness")

    def test_normal_english_negative(self):
        text = "Worst customer service ever, completely terrible and horrible experience."
        res = analyze_single_post(text)
        self.assertEqual(res["sentiment"], "Negative")
        self.assertLess(res["sentiment_score"], -0.2)
        self.assertIn(res["emotion"], ["Anger", "Frustration"])

    def test_neutral_text(self):
        text = "The annual meeting is scheduled tomorrow at 10 AM in Conference Room B."
        res = analyze_single_post(text)
        self.assertEqual(res["sentiment"], "Neutral")
        self.assertEqual(res["emotion"], "Neutral")

    def test_hinglish_positive(self):
        text = "Yeh biryani ekdam mast aur zabardast hai! Bohot maza aaya."
        res = analyze_single_post(text)
        self.assertEqual(res["sentiment"], "Positive")
        self.assertEqual(res["emotion"], "Happiness")
        self.assertEqual(res["topic"], "Food")

    def test_hinglish_negative(self):
        text = "Bilkul bekaar service thi, dimag kharab ho gaya pura din."
        res = analyze_single_post(text)
        self.assertEqual(res["sentiment"], "Negative")
        self.assertIn(res["emotion"], ["Frustration", "Anger"])

    def test_negation_handling_english(self):
        # "not good" should flip positive word to negative
        text = "The software update is not good at all."
        res = analyze_sentiment(text)
        self.assertEqual(res["label"], "Negative")

        # "not bad" should not be classified as negative
        text_not_bad = "This solution is not bad at all, actually works."
        res_not_bad = analyze_sentiment(text_not_bad)
        self.assertIn(res_not_bad["label"], ["Positive", "Neutral"])

    def test_negation_handling_hinglish(self):
        # "accha nahi hai" -> negative
        text = "Yeh phone accha nahi hai."
        res = analyze_sentiment(text)
        self.assertEqual(res["label"], "Negative")

    def test_repeated_characters_normalization(self):
        text = "sooooo goooood mastttt"
        normalized = normalize_repeated_characters(text)
        # 3+ characters reduced to 2
        self.assertEqual(normalized, "soo good mastt")

    def test_url_removal(self):
        text = "Check this cool app out: https://example.com/download and www.test.org"
        cleaned = clean_text(text)
        self.assertNotIn("http", cleaned)
        self.assertNotIn("example.com", cleaned)
        self.assertNotIn("www", cleaned)

    def test_mention_extraction(self):
        text = "Hey @elonmusk and @satyanadella check this out"
        mentions = extract_mentions(text)
        self.assertEqual(mentions, ["elonmusk", "satyanadella"])

    def test_hashtag_extraction_and_splitting(self):
        text = "Stuck in #BangaloreTraffic again! #TechNews"
        hashtags = extract_hashtags(text)
        self.assertEqual(hashtags, ["bangaloretraffic", "technews"])

        split_tag = split_hashtag_words("BangaloreTraffic")
        self.assertEqual(split_tag, "bangalore traffic")

    def test_emoji_extraction_and_scoring(self):
        text_happy = "Loved the dinner ❤️ 😊"
        res_happy = analyze_single_post(text_happy)
        self.assertEqual(res_happy["sentiment"], "Positive")
        self.assertEqual(res_happy["emotion"], "Happiness")

        text_angry = "Terrible delay! 😡 🤬"
        res_angry = analyze_single_post(text_angry)
        self.assertEqual(res_angry["sentiment"], "Negative")
        self.assertEqual(res_angry["emotion"], "Anger")

    def test_empty_and_none_text_safety(self):
        # Must never crash on None
        res_none = analyze_single_post(None)
        self.assertEqual(res_none["sentiment"], "Neutral")
        self.assertEqual(res_none["emotion"], "Neutral")
        self.assertEqual(res_none["topic"], "Other")

        # Must never crash on empty string or whitespace
        res_empty = analyze_single_post("    ")
        self.assertEqual(res_empty["sentiment"], "Neutral")
        self.assertEqual(res_empty["emotion"], "Neutral")

        # Emoji-only string
        res_emoji = analyze_single_post("🎉 🚀")
        self.assertEqual(res_emoji["sentiment"], "Positive")
        self.assertEqual(res_emoji["emotion"], "Happiness")

    def test_missing_and_malformed_timestamp(self):
        res_missing = analyze_single_post("Good morning", timestamp=None)
        self.assertTrue(len(res_missing["timestamp"]) > 0)

        res_malformed = analyze_single_post("Good evening", timestamp="invalid-date-format")
        self.assertTrue(len(res_malformed["timestamp"]) > 0)

    def test_topic_classification(self):
        res_traffic = classify_topic("Outer Ring Road has 5 km traffic jam due to signal fault")
        self.assertEqual(res_traffic["topic"], "Traffic")

        res_tech = classify_topic("Python open-source machine learning model release")
        self.assertEqual(res_tech["topic"], "Technology")

        res_cricket = classify_topic("Kohli scored 100 runs in the IPL final match!")
        self.assertEqual(res_cricket["topic"], "Sports")

    def test_dataframe_analysis_and_aggregations(self):
        df = pd.DataFrame([
            {"post_id": 1, "timestamp": "2026-09-01 10:00:00", "author": "u1", "text": "Loved it! Mast tha ❤️"},
            {"post_id": 2, "timestamp": "2026-09-01 11:00:00", "author": "u2", "text": "Horrible traffic jam 😡"},
            {"post_id": 3, "timestamp": "2026-09-01 12:00:00", "author": "u3", "text": "Office meeting at 2 PM."}
        ])

        analyzed_df = analyze_dataframe(df)
        self.assertEqual(len(analyzed_df), 3)
        self.assertIn("sentiment", analyzed_df.columns)
        self.assertIn("emotion", analyzed_df.columns)
        self.assertIn("topic", analyzed_df.columns)

        summary = get_sentiment_summary(analyzed_df)
        self.assertEqual(summary["total_posts"], 3)
        self.assertEqual(summary["positive_count"], 1)
        self.assertEqual(summary["negative_count"], 1)
        self.assertEqual(summary["neutral_count"], 1)

        emo_summary = get_emotion_summary(analyzed_df)
        self.assertFalse(emo_summary.empty)

        topic_summary = get_topic_summary(analyzed_df)
        self.assertFalse(topic_summary.empty)

        time_df = get_sentiment_over_time(analyzed_df)
        self.assertFalse(time_df.empty)

    def test_input_sanitization_and_upload_safety(self):
        # HTML tag sanitization
        dirty = "<script>alert('hack')</script>Normal text"
        clean = sanitize_input_text(dirty)
        self.assertNotIn("<script>", clean)
        self.assertIn("Normal text", clean)

        # File validation with mock file
        class MockFile:
            def __init__(self, name, content):
                self.name = name
                self._content = content.encode("utf-8")
                self.size = len(self._content)

            def getvalue(self):
                return self._content

        valid_csv = MockFile("test.csv", "text,author\nAwesome product!,alice\n")
        ok, msg, parsed = validate_uploaded_file(valid_csv)
        self.assertTrue(ok)
        self.assertIsNotNone(parsed)

    def test_database_operations(self):
        from src.database import init_db, insert_post, insert_posts_batch, get_all_posts, get_db_stats
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmp_dir:
            test_db = os.path.join(tmp_dir, "test_socialpulse.db")
            
            # 1. Initialize empty DB
            ok, msg = init_db(db_path=test_db, seed_from_csv=False)
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(test_db))

            # 2. Insert single post with parameterized query
            sample_record = {
                "post_id": 999,
                "timestamp": "2026-09-11 12:00:00",
                "author": "db_tester",
                "raw_text": "Testing database storage #Tech",
                "cleaned_text": "testing database storage tech",
                "source": "unit_test",
                "sentiment": "Positive",
                "sentiment_score": 0.85,
                "sentiment_confidence": 0.90,
                "emotion": "Happiness",
                "emotion_confidence": 0.88,
                "topic": "Technology",
                "topic_confidence": 0.92,
                "hashtags": ["Tech"],
                "mentions": [],
                "keywords": ["database", "storage"]
            }
            row_id = insert_post(sample_record, db_path=test_db)
            self.assertIsNotNone(row_id)

            # 3. Retrieve and verify
            df = get_all_posts(db_path=test_db)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.iloc[0]["author"], "db_tester")
            self.assertEqual(df.iloc[0]["sentiment"], "Positive")
            self.assertEqual(df.iloc[0]["topic"], "Technology")

            # 4. SQL Injection safety test in text and author
            injection_record = {
                "post_id": 1000,
                "timestamp": "2026-09-11 12:05:00",
                "author": "hacker'; DROP TABLE posts; --",
                "raw_text": "Malicious payload'); DELETE FROM posts; --",
                "cleaned_text": "malicious payload delete from posts",
                "source": "injection_test",
                "sentiment": "Neutral",
                "sentiment_score": 0.0,
                "sentiment_confidence": 0.5,
                "emotion": "Neutral",
                "emotion_confidence": 0.5,
                "topic": "Other",
                "topic_confidence": 0.5,
                "hashtags": [],
                "mentions": [],
                "keywords": []
            }
            inj_id = insert_post(injection_record, db_path=test_db)
            self.assertIsNotNone(inj_id)

            # Confirm table is NOT dropped and still has 2 records
            df_after = get_all_posts(db_path=test_db)
            self.assertEqual(len(df_after), 2)

            # 5. DB Stats
            stats = get_db_stats(db_path=test_db)
            self.assertTrue(stats["exists"])
            self.assertEqual(stats["total_posts"], 2)


if __name__ == "__main__":
    unittest.main()

