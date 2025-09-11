#!/usr/bin/env python3
"""
Preprocess Reddit CSV for Hive/Spark sentiment & trend analysis.

What it does
- Chooses text: selftext if present else title  → text_raw
- Cleans text: strip URLs/HTML/emoji/punct, lowercase, normalize spaces → text_clean
- Adds word_count
- Converts created_utc → created_at_utc (ISO) and local time (default Asia/Bangkok)
- Adds dt partition column (YYYY-MM-DD based on UTC)
- Drops NSFW rows (over_18 == True) unless --keep-nsfw
- Drops empty text rows
- Deduplicates by (id, permalink, url, title) keeping the newest
- Writes CSV (optionally Parquet with --parquet)

Usage
  python preprocess_reddit.py -i posts_clean.csv -o posts_preprocessed.csv
  python preprocess_reddit.py -i posts_clean.csv -o out/ --parquet   # writes Parquet folder
"""

import argparse
import re
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

# -----------------------
# Regexes for cleaning
# -----------------------
URL_RE   = re.compile(r"https?://\S+|www\.\S+")
HTML_RE  = re.compile(r"<[^>]+>")
EMOJI_RE = re.compile(
    "["                     # common emoji blocks
    "\U0001F600-\U0001F64F" # emoticons
    "\U0001F300-\U0001F5FF" # symbols & pictographs
    "\U0001F680-\U0001F6FF" # transport & map
    "\U0001F1E0-\U0001F1FF" # flags
    "\U00002702-\U000027B0" # dingbats
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
PUNCT_RE = re.compile(r"[^\w\s]")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u200b", " ")          # zero-width spaces
    s = URL_RE.sub(" ", s)
    s = HTML_RE.sub(" ", s)
    s = EMOJI_RE.sub(" ", s)
    s = PUNCT_RE.sub(" ", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def to_datetime_utc(v):
    try:
        # Reddit 'created_utc' is seconds since epoch
        return datetime.fromtimestamp(float(v), tz=timezone.utc)
    except Exception:
        return pd.NaT

def get_local_tz(tz_str: str, offset_hours: int):
    """
    Try Python 3.9+ zoneinfo; fallback to fixed offset if unavailable.
    """
    try:
        from zoneinfo import ZoneInfo  # py3.9+
        return ZoneInfo(tz_str)
    except Exception:
        return timezone(timedelta(hours=offset_hours))

def preprocess(
    df: pd.DataFrame,
    keep_nsfw: bool = False,
    tz_str: str = "Asia/Bangkok",
    offset_hours: int = 7,
):
    # ----- choose a text field -----
    has_self = "selftext" in df.columns
    has_title = "title" in df.columns
    if not (has_self or has_title):
        raise ValueError("Input must have at least one of: 'selftext' or 'title'.")

    selftext = df["selftext"].astype(str) if has_self else pd.Series([""]*len(df))
    title    = df["title"].astype(str)    if has_title else pd.Series([""]*len(df))

    text_raw = np.where(
        (has_self and selftext.str.strip() != ""),
        selftext,
        title,
    )
    df["text_raw"] = text_raw
    df["text_clean"] = [clean_text(t) for t in text_raw]
    df["word_count"] = df["text_clean"].str.split().str.len().fillna(0).astype(int)

    # ----- timestamps -----
    if "created_utc" in df.columns:
        dt_utc = df["created_utc"].apply(to_datetime_utc)
    else:
        # fallback: find a column that looks like created_utc
        cand = next((c for c in df.columns if "created" in c and "utc" in c), None)
        if cand is None:
            dt_utc = pd.Series([pd.NaT] * len(df))
        else:
            dt_utc = df[cand].apply(to_datetime_utc)

    df["created_at_utc"] = dt_utc

    local_tz = get_local_tz(tz_str, offset_hours)
    df["created_at_local"] = df["created_at_utc"].apply(
        lambda d: (pd.NaT if pd.isna(d) else d.astimezone(local_tz))
    )
    # partition date (UTC for stability)
    df["dt"] = df["created_at_utc"].dt.strftime("%Y-%m-%d")

    # ----- basic filters -----
    if not keep_nsfw and "over_18" in df.columns:
        df = df[df["over_18"] == False].copy()

    df = df[df["text_clean"].str.len() > 0].copy()

    # ----- dedupe -----
    keys = [c for c in ["id", "permalink", "url", "title"] if c in df.columns]
    if keys:
        df = df.sort_values(by=["created_at_utc"], ascending=False)\
               .drop_duplicates(subset=keys, keep="first")

    # ----- keep useful columns if present -----
    keep_cols = [
        "id", "subreddit", "author",
        "title", "selftext", "text_raw", "text_clean", "word_count",
        "url", "permalink", "link_flair_text",
        "ups", "downs", "score", "num_comments", "upvote_ratio",
        "is_self", "over_18", "spoiler", "stickied", "locked",
        "created_utc", "created_at_utc", "created_at_local", "dt"
    ]
    existing = [c for c in keep_cols if c in df.columns]
    return df[existing].reset_index(drop=True)

def main():
    p = argparse.ArgumentParser(description="Preprocess Reddit CSV for Hive/Spark.")
    p.add_argument("-i", "--input", required=True, help="Input CSV path")
    p.add_argument("-o", "--output", required=True,
                   help="Output CSV file path OR directory if --parquet")
    p.add_argument("--keep-nsfw", action="store_true", help="Keep NSFW posts (default: drop)")
    p.add_argument("--timezone", default="Asia/Bangkok",
                   help="IANA timezone name (default: Asia/Bangkok)")
    p.add_argument("--offset", type=int, default=7,
                   help="Fallback UTC offset hours if zoneinfo unavailable (default: 7)")
    p.add_argument("--parquet", action="store_true",
                   help="Write Parquet instead of CSV (output treated as directory)")
    p.add_argument("--encoding", default="utf-8",
                   help="CSV encoding (default: utf-8)")
    p.add_argument("--sep", default=",", help="CSV delimiter (default: ,)")
    args = p.parse_args()

    try:
        df = pd.read_csv(args.input, encoding=args.encoding, sep=args.sep, on_bad_lines="skip")
    except TypeError:
        # pandas < 1.4 compatibility (no on_bad_lines)
        df = pd.read_csv(args.input, encoding=args.encoding, sep=args.sep, error_bad_lines=False)

    out = preprocess(
        df,
        keep_nsfw=args.keep_nsfw,
        tz_str=args.timezone,
        offset_hours=args.offset,
    )

    if args.parquet:
        # write parquet; args.output is a folder
        out.to_parquet(args.output, index=False)
        print(f"[write] parquet → {args.output} (rows={len(out)})")
    else:
        out.to_csv(args.output, index=False)
        print(f"[write] csv → {args.output} (rows={len(out)})")

if __name__ == "__main__":
    # Optional: make pandas printing predictable if user runs interactively
    pd.set_option("display.max_colwidth", 200)
    main()
