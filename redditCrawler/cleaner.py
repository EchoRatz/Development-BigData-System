#!/usr/bin/env python3
"""
Clean Reddit post exports from the crawler.

Features:
- Handles 1+ CSV inputs (glob patterns supported)
- Converts created_utc -> created_at_utc (ISO8601, Z)
- Adds 'date' (YYYY-MM-DD)
- Trims/normalizes text fields
- Drops duplicates (by id, then permalink)
- Optional: filter by date / subreddit / min score
- Optional: anonymize authors (salted hash)
- Outputs CSV (default) or Parquet

Usage examples:
  # Clean one file to CSV (default)
  python clean_reddit_csv.py out/posts.part001.csv -o out/posts_clean.csv

  # Clean multiple parts, write Parquet, anonymize authors
  python clean_reddit_csv.py "out/posts.part*.csv" -o out/posts_clean.parquet --format parquet --anonymize-authors --salt my_secret_salt

  # Filter to Jan–Sept 2025 and r/formula1 only, min score 1
  python clean_reddit_csv.py "out/posts.part*.csv" -o out/f1_clean.csv \
      --since 2025-01-01 --until 2025-09-11 --subreddits formula1 --min-score 1
"""
import argparse
import glob
import hashlib
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd


def _to_iso_utc(epoch):
    try:
        if pd.isna(epoch):
            return None
        return datetime.fromtimestamp(float(epoch), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return None


def _strip_or_none(x):
    if pd.isna(x):
        return None
    s = str(x).replace("\x00", "").strip()
    # collapse excessive whitespace inside text-y fields but keep single spaces
    return " ".join(s.split())


def _hash_author(author: Optional[str], salt: str) -> Optional[str]:
    if not author or pd.isna(author):
        return None
    h = hashlib.sha256((salt + ":" + str(author)).encode("utf-8")).hexdigest()
    return f"anon_{h[:12]}"


def _read_many(paths: List[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__source_file"] = os.path.basename(p)
            frames.append(df)
        except FileNotFoundError:
            print(f"[warn] file not found: {p}", file=sys.stderr)
        except Exception as e:
            print(f"[warn] failed to read {p}: {e}", file=sys.stderr)
    if not frames:
        raise SystemExit("[error] no input files were read.")
    return pd.concat(frames, ignore_index=True)


def clean_posts(df: pd.DataFrame,
                since: Optional[str],
                until: Optional[str],
                subs: Optional[List[str]],
                min_score: Optional[int],
                anonymize_authors: bool,
                salt: str) -> pd.DataFrame:
    before = len(df)

    # Normalize key columns if present
    for col in ["title", "selftext", "url", "permalink", "author", "subreddit", "link_flair_text"]:
        if col in df.columns:
            df[col] = df[col].map(_strip_or_none)

    # Convert created_utc -> created_at_utc (ISO Z) + date
    if "created_utc" in df.columns:
        df["created_at_utc"] = df["created_utc"].map(_to_iso_utc)
        # derive date for easy grouping
        df["date"] = pd.to_datetime(df["created_at_utc"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        df["created_at_utc"] = None
        df["date"] = None

    # Filters: date range (inclusive)
    if since:
        df = df[pd.to_datetime(df["created_at_utc"], errors="coerce") >= pd.to_datetime(since).tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT", errors="coerce")]
    if until:
        # include the whole day if only a date was supplied
        if len(until) == 10:  # YYYY-MM-DD
            until_ts = pd.to_datetime(until + "T23:59:59Z")
        else:
            until_ts = pd.to_datetime(until)
        df = df[pd.to_datetime(df["created_at_utc"], errors="coerce") <= until_ts]

    # Filter by subreddit(s)
    if subs:
        low = {s.lower() for s in subs}
        if "subreddit" in df.columns:
            df = df[df["subreddit"].str.lower().isin(low)]

    # Filter by min score
    if min_score is not None and "score" in df.columns:
        df = df[pd.to_numeric(df["score"], errors="coerce").fillna(-10**9) >= min_score]

    # Drop rows with no visible content (both title and selftext empty)
    if "title" in df.columns and "selftext" in df.columns:
        df = df[~(df["title"].isna() & df["selftext"].isna())]

    # Deduplicate: prefer unique post id; fallback to permalink
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="first")
    if "permalink" in df.columns:
        df = df.drop_duplicates(subset=["permalink"], keep="first")

    # Anonymize authors if requested
    if anonymize_authors and "author" in df.columns:
        df["author"] = df["author"].map(lambda a: _hash_author(a, salt))

    # Reorder columns (if present)
    preferred = [
        "kind", "id", "subreddit", "author",
        "title", "selftext",
        "score", "ups", "downs", "num_comments", "upvote_ratio",
        "link_flair_text", "over_18", "spoiler", "stickied", "locked",
        "url", "permalink",
        "created_utc", "created_at_utc", "date", "edited"
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]

    after = len(df)
    print(f"[summary] rows in: {before}  →  rows out: {after}  (dropped {before - after})")
    return df


def main():
    ap = argparse.ArgumentParser(description="Clean Reddit crawler CSV exports.")
    ap.add_argument("inputs", nargs="+", help="Input CSV file(s) or glob pattern(s). e.g. out/posts.part001.csv or 'out/posts.part*.csv'")
    ap.add_argument("-o", "--output", required=True, help="Output file path (e.g., out/posts_clean.csv or .parquet)")
    ap.add_argument("--format", choices=["csv", "parquet"], default=None, help="Force output format (default inferred from extension)")
    ap.add_argument("--since", help="Keep rows on/after this date/time (YYYY-MM-DD or ISO8601)", default=None)
    ap.add_argument("--until", help="Keep rows on/before this date/time (YYYY-MM-DD or ISO8601)", default=None)
    ap.add_argument("--subreddits", nargs="*", help="Filter to these subreddits (names without r/)", default=None)
    ap.add_argument("--min-score", type=int, help="Keep only rows with score >= N", default=None)
    ap.add_argument("--anonymize-authors", action="store_true", help="Replace author with salted hash")
    ap.add_argument("--salt", default="change_me_salt", help="Salt used when anonymizing authors")
    args = ap.parse_args()

    # Expand globs
    paths: List[str] = []
    for pat in args.inputs:
        matched = glob.glob(pat)
        if not matched:
            print(f"[warn] no files matched pattern: {pat}", file=sys.stderr)
        paths.extend(matched)
    if not paths:
        raise SystemExit("[error] no input files found.")

    df = _read_many(paths)

    df = clean_posts(
        df,
        since=args.since,
        until=args.until,
        subs=args.subreddits,
        min_score=args.min_score,
        anonymize_authors=args.anonymize_authors,
        salt=args.salt,
    )

    # Decide output format
    out_fmt = (args.format or os.path.splitext(args.output)[1].lower().lstrip(".") or "csv")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if out_fmt == "parquet":
        df.to_parquet(args.output, index=False)
    elif out_fmt == "csv":
        df.to_csv(args.output, index=False)
    else:
        raise SystemExit(f"[error] unsupported output format: {out_fmt}")

    print(f"[write] → {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
