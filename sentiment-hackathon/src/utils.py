"""Utility functions for SocialPulse.

Includes:
- Safe CSV file validation (extension, size, content structure)
- Configurable rate limiting for API/interactive actions
- Sanitization and security safeguards
- Safe data loader
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

# Security settings
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".csv"}
MAX_ROW_COUNT = 10000


class ActionRateLimiter:
    """Configurable in-memory sliding-window rate limiter for interactive actions."""

    def __init__(self, max_requests: int = 30, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.timestamps: List[float] = []

    def allow_request(self) -> Tuple[bool, str]:
        """Check if request is allowed under current rate limits."""
        now = time.time()
        # Clean expired timestamps
        self.timestamps = [ts for ts in self.timestamps if now - ts < self.window_seconds]

        if len(self.timestamps) >= self.max_requests:
            remaining = int(self.window_seconds - (now - self.timestamps[0]))
            return False, f"Rate limit exceeded. Please wait {remaining} seconds."

        self.timestamps.append(now)
        return True, "Allowed"


def validate_uploaded_file(uploaded_file: Any) -> Tuple[bool, Optional[str], Optional[pd.DataFrame]]:
    """Validate uploaded CSV file for security, format, and structure.
    
    Checks:
    - Non-null and valid object
    - File extension is strictly .csv
    - File size is within 5MB limit
    - Content is parseable CSV
    - Text column presence or inferability
    - Row count within safety bounds
    """
    if uploaded_file is None:
        return False, "No file provided.", None

    filename = getattr(uploaded_file, "name", "uploaded.csv")
    if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return False, f"Unsupported file type. Only CSV files ({', '.join(ALLOWED_EXTENSIONS)}) are permitted.", None

    # Check file size
    file_size = getattr(uploaded_file, "size", None)
    if file_size is not None and file_size > MAX_FILE_SIZE_BYTES:
        return False, f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB.", None

    try:
        # Read content safely
        content = uploaded_file.getvalue()
        if len(content) > MAX_FILE_SIZE_BYTES:
            return False, "File content exceeds size limit.", None

        # Parse CSV safely with pandas
        df = pd.read_csv(io.BytesIO(content), nrows=MAX_ROW_COUNT + 1)

        if df.empty:
            return False, "The uploaded CSV is empty.", None

        if len(df) > MAX_ROW_COUNT:
            return False, f"CSV contains too many rows. Maximum allowed is {MAX_ROW_COUNT}.", None

        # Verify there is at least one usable string/text column
        text_cols = [c for c in df.columns if df[c].dtype == object]
        if not text_cols and len(df.columns) > 0:
            # Check if first column can be cast to string
            df[df.columns[0]] = df[df.columns[0]].astype(str)
            text_cols = [df.columns[0]]

        if not text_cols:
            return False, "No textual columns found in the uploaded CSV.", None

        return True, None, df

    except Exception as e:
        # Avoid leaking internal file paths or raw exceptions to user
        return False, "Unable to parse CSV file. Ensure the file is valid UTF-8 formatted CSV.", None


def sanitize_input_text(raw_input: Any, max_length: int = 1000) -> str:
    """Sanitize user input string against HTML injection and excessive length."""
    if raw_input is None:
        return ""
    text = str(raw_input).strip()
    # Strip potential script tags or HTML tags
    text = text.replace("<script>", "").replace("</script>", "")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    return text[:max_length]
