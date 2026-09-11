# SocialPulse: AI Sentiment & Topic Intelligence Dashboard

SocialPulse is an AI-powered social media sentiment, emotion, topic, and trend analysis dashboard designed for social media analytics. It provides full support for English and Hinglish (code-mixed), negations, emojis, @mentions, URLs, and repeated characters.

---

## Project Structure

```
sentiment-hackathon/
├── app.py                      # Main Streamlit Dashboard Application
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation and demo guide
├── .gitignore                  # Git secrets and cache ignore patterns
├── favicon.png                 # Application brand icon
├── robots.txt                  # Search crawler configuration
├── sitemap.xml                 # Canonical XML sitemap
├── llms.txt                    # Machine-readable documentation
├── data/
│   └── sample_posts.csv        # 110 realistic fictional social posts (English + Hinglish)
├── src/
│   ├── __init__.py             # Package marker
│   ├── preprocessing.py        # Robust text cleaner, Hinglish preservation, emoji/tag extractor
│   ├── sentiment.py            # Transparent lexical engine with bidirectional negation window
│   ├── emotion.py              # 7-class emotion classifier with confidence calibration
│   ├── topics.py               # 11 topic categories + TF-IDF keyword extraction
│   ├── analysis.py             # Batch processing, KPI aggregations, timeline & spike detection
│   └── utils.py                # Safe CSV validator, rate limiter, input sanitization
└── tests/
    └── test_pipeline.py        # Automated test suite (17 comprehensive tests)
```

---

## Features Implemented

1. **Multilingual & Code-Mixed NLP Pipeline**:
   - Robust processing of English and Hinglish ("mast", "zabardast", "bekaar", "dimag kharab", "bohot badiya", "ghatiya").
   - Bidirectional negation handling supporting both English ("not good", "never again") and Hindi Subject-Object-Verb syntax ("accha nahi", "pasand nahi").
   - Character repetition normalization ("sooo goood" -> "soo good", "mastttt" -> "mastt").
   - URL removal and @mention extraction.
   - Comprehensive emoji lexicon (mapping ❤️, 😡, 😭, 😤, 🚀, etc. to sentiment and emotions).

2. **Sentiment Analysis**:
   - Classes: `Positive`, `Negative`, `Neutral`.
   - Continuous sentiment score (-1.0 to +1.0) and calibrated confidence (0.50 to 0.98).

3. **Emotion Classification**:
   - 7 Target Emotions: `Happiness`, `Anger`, `Sadness`, `Frustration`, `Fear`, `Surprise`, `Neutral`.
   - Multi-class scoring system with negation suppression.

4. **Topic & Keyword Extraction**:
   - 11 Categories: Technology, Sports, Food, Travel, Education, Traffic, Entertainment, Shopping, Customer Service, Politics, Other.
   - TF-IDF corpus keyword extraction and hashtag frequency metrics.

5. **Analytical Streamlit Dashboard**:
   - Executive KPI Cards: Total Posts, Positive %, Negative %, Neutral %, Top Topic (all dynamically calculated).
   - Plotly Visualizations: Sentiment donut chart, emotion bar chart, timeline trend line chart, horizontal topic volume chart, keyword/hashtag bars.
   - Post Explorer: Search by keyword/user, filter by sentiment, emotion, and topic, with CSV export.
   - Interactive Single Post Analyzer: Immediate testing of any typed text.

6. **Live Simulation Mode**:
   - Live stream runner that injects synthetic social posts in real time.
   - Automatically recalculates all KPIs, charts, and displays the latest analyzed post.
   - Clearly labeled as a synthetic simulation.

7. **Security & Governance**:
   - Strict CSV upload validation (file extension, 5MB size limit, content check).
   - Configurable rate limiting for interactive actions.
   - Input sanitization against script injection.
   - Built-in Privacy Policy and Terms and Conditions views.

---

## Installation

Ensure Python 3.11+ is installed.

```bash
# Clone or navigate to the project directory
cd "project 2"

# Install dependencies
python -m pip install -r requirements.txt
```

---

## Running the Application

To launch the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## Running Automated Tests

Run the comprehensive unit test suite:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

All 17 tests validate English/Hinglish sentiment, SOV negation, emoji extraction, repeated character normalization, safe handling of `None` and empty strings, and file upload safety.

---

## Suggested 2-Minute Demo Flow

1. **Executive Overview (0:00 - 0:30)**:
   - Point out the 5 KPI cards at the top (Total Posts, Positive %, Negative %, Neutral %, Top Topic).
   - Show that all metrics are calculated from real post data.
   - Highlight the Sentiment Donut Chart, Emotion Breakdown (Happiness, Frustration, Anger, etc.), and Sentiment Trend over time.

2. **Post Explorer & Filtering (0:30 - 0:50)**:
   - Switch to the "Post Explorer" tab.
   - Filter by Sentiment (e.g. "Negative") and Topic (e.g. "Traffic" or "Customer Service").
   - Notice posts with Bangalore traffic complaints and customer support issues correctly tagged.
   - Demonstrate the free-text search (e.g., search "Silk Board" or "Zomato").

3. **Interactive Single Post Analyzer (0:50 - 1:20)**:
   - Switch to the "Single Post Analyzer" tab.
   - Paste a Hinglish sentence:
     `"Silk board traffic was completely jammed today! Bilkul bekaar experience. #BangaloreTraffic"`
   - Click "Analyze Post": show that Sentiment is Negative, Emotion is Frustration, Topic is Traffic, and Hashtag is extracted.
   - Try a post with negation:
     `"Yeh phone bilkul accha nahi hai, paise barbad ho gaye."`
     Notice it detects Negative sentiment with Frustration/Anger.
   - Try an English post:
     `"Incredible rocket launch today! Absolutely mindblowing engineering. #SpaceTech 🚀"`
     Notice Positive sentiment, Happiness emotion, Technology topic.

4. **Live Simulation (1:20 - 1:50)**:
   - Toggle "Enable Live Simulation" in the sidebar.
   - Watch incoming synthetic social media posts appear in real time with immediate KPI and chart updates.
   - Point out the transparent "LIVE SIMULATION" notice.

5. **Compliance & Legal (1:50 - 2:00)**:
   - View the "Compliance & Policies" tab displaying the Privacy Policy, Terms of Service, and data protection practices.

---

## Known Limitations

- The lexical-heuristic model is transparent, deterministic, and extremely fast, but highly subtle figurative sarcasm (without clear lexical or negation cues) may classify as neutral.
- Hinglish transliteration variants have broad coverage, though regional slang with novel phonetic spellings can be further expanded in the lexicon dictionary.
