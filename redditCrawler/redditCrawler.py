#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
import pathlib
import datetime as dt
from typing import List, Dict, Any, Optional, Union

import praw
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

# ---------------- Utils ----------------

def iso_to_epoch(iso_str: str) -> Optional[int]:
    if not iso_str:
        return None
    return int(dt.datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp())

def to_safe_int(x, default=None):
    try:
        return int(x)
    except Exception:
        return default

def ensure_dir(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def normalize_depth(cfg_val: Union[str, int, None]) -> Optional[int]:
    """
    Returns:
      None -> means "all"
      1    -> only top-level comments
      2    -> top-level + one level deep
      ...
    """
    if cfg_val is None:
        return None
    if isinstance(cfg_val, str):
        if cfg_val.strip().lower() == "all":
            return None
        try:
            return int(cfg_val)
        except Exception:
            return None
    if isinstance(cfg_val, int):
        if cfg_val <= 0:
            return 1
        return cfg_val
    return None

# ---------------- Output Writers ----------------

class RotatingWriter:
    """
    Supports CSV and NDJSON ("json") with file rotation.
    - CSV: writes header once per part; fields = union of seen keys so far in that part.
    - JSON: newline-delimited JSON (one object per line).
    """
    def __init__(self, out_dir: str, base_name: str, fmt: str, rotate_every: int):
        self.out_dir = out_dir
        self.base_name = base_name
        self.fmt = fmt.lower()
        self.rotate_every = rotate_every or 10_000
        self.buffer: List[Dict[str, Any]] = []
        self.part = 0
        self.csv_fields: List[str] = []
        ensure_dir(out_dir)

        if self.fmt not in ("csv", "json"):
            raise ValueError("output 'format' must be 'csv' or 'json'")

    def _next_path(self) -> str:
        ext = "csv" if self.fmt == "csv" else "json"
        filename = f"{self.base_name}.part{self.part:03d}.{ext}"
        return os.path.join(self.out_dir, filename)

    def add(self, row: Dict[str, Any]):
        self.buffer.append(row)
        if len(self.buffer) >= self.rotate_every:
            self.flush()

    def _write_csv(self, path: str, rows: List[Dict[str, Any]]):
        # Build/extend field list for this part
        for r in rows:
            for k in r.keys():
                if k not in self.csv_fields:
                    self.csv_fields.append(k)
        df = pd.DataFrame(rows, columns=self.csv_fields)
        header = True  # new part => write header
        df.to_csv(path, index=False, header=header, mode="w", encoding="utf-8")

    def _write_ndjson(self, path: str, rows: List[Dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as f:
            for obj in rows:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def flush(self, final: bool = False):
        if not self.buffer:
            return
        self.part += 1
        fpath = self._next_path()
        if self.fmt == "csv":
            # reset fields for each part
            self.csv_fields = []
            self._write_csv(fpath, self.buffer)
        else:
            self._write_ndjson(fpath, self.buffer)
        self.buffer.clear()
        if final:
            print(f"[write] → {fpath}")

    def close(self):
        self.flush(final=True)

# ---------------- Reddit Client ----------------

class RedditClient:
    def __init__(self):
        load_dotenv()
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            username=os.getenv("REDDIT_USERNAME"),
            password=os.getenv("REDDIT_PASSWORD"),
            user_agent=os.getenv("REDDIT_USER_AGENT", "reddit-crawler")
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(Exception),
    )
    def search_submissions(self, subreddit: str, query: str, since: Optional[int], until: Optional[int]):
        # Use CloudSearch syntax with timestamp filter
        time_query = ""
        if since and until:
            time_query = f"timestamp:{since}..{until}"
        elif since:
            time_query = f"timestamp:{since}..{int(time.time())}"
        elif until:
            time_query = f"timestamp:0..{until}"

        final_query = " ".join(x for x in [query, time_query] if x).strip()
        return self.reddit.subreddit(subreddit).search(
            query=final_query,
            sort="new",
            syntax="cloudsearch",
            limit=None
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
        retry=retry_if_exception_type(Exception),
    )
    def new_submissions(self, subreddit: str):
        return self.reddit.subreddit(subreddit).new(limit=None)

# ---------------- Flatteners ----------------

def flatten_submission(s) -> Dict[str, Any]:
    return {
        "kind": "submission",
        "id": s.id,
        "subreddit": str(s.subreddit),
        "author": str(s.author) if s.author else None,
        "title": s.title,
        "selftext": s.selftext,
        "url": s.url,
        "is_self": s.is_self,
        "over_18": getattr(s, "over_18", None),
        "spoiler": getattr(s, "spoiler", None),
        "stickied": getattr(s, "stickied", None),
        "locked": getattr(s, "locked", None),
        "upvote_ratio": getattr(s, "upvote_ratio", None),
        "ups": getattr(s, "ups", None),
        "downs": getattr(s, "downs", None) if hasattr(s, "downs") else None,
        "score": getattr(s, "score", None),
        "num_comments": getattr(s, "num_comments", None),
        "created_utc": to_safe_int(getattr(s, "created_utc", None)),
        "link_flair_text": getattr(s, "link_flair_text", None),
        "edited": s.edited if isinstance(getattr(s, "edited", False), bool) else to_safe_int(getattr(s, "edited", None)),
        "permalink": f"https://www.reddit.com{s.permalink}" if getattr(s, "permalink", None) else None,
    }

def flatten_comment(c, submission_id: str, subreddit: str) -> Dict[str, Any]:
    return {
        "kind": "comment",
        "id": c.id,
        "subreddit": subreddit,
        "submission_id": submission_id,
        "author": str(c.author) if c.author else None,
        "body": c.body,
        "score": getattr(c, "score", None),
        "created_utc": to_safe_int(getattr(c, "created_utc", None)),
        "is_submitter": getattr(c, "is_submitter", None),
        "parent_id": getattr(c, "parent_id", None),
        "permalink": f"https://www.reddit.com{getattr(c, 'permalink', '')}",
        "depth": getattr(c, "depth", None),
    }

# ---------------- Comment Crawling ----------------

def crawl_comments_for_submission(submission, max_comments: int, sleep_ms: int, depth_limit: Optional[int]) -> List[Dict[str, Any]]:
    """
    depth_limit:
      None -> all
      1    -> only top-level (depth==0)
      2    -> <=1, etc.
    """
    try:
        # expand all "MoreComments" so .list() returns a flat list with .depth set
        submission.comments.replace_more(limit=0)
    except Exception:
        pass

    comments_data: List[Dict[str, Any]] = []
    count = 0
    for c in submission.comments.list():
        d = getattr(c, "depth", 0)

        # apply depth filter (top-level has depth==0)
        if depth_limit is not None:
            allowed_max_depth = depth_limit - 1  # depth_limit=1 -> allow depth 0 only
            if d is not None and d > allowed_max_depth:
                continue

        comments_data.append(flatten_comment(c, submission.id, str(submission.subreddit)))
        count += 1

        if max_comments and count >= max_comments:
            break
        if sleep_ms:
            time.sleep(sleep_ms / 1000.0)

    return comments_data

# ---------------- Main Runner ----------------

def run(args):
    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f)
        
    import os, time, json
    config_path = os.path.abspath(args.config)
    try:
        st = os.stat(config_path)
        print(f"[CONFIG] Using: {config_path} (size={st.st_size}B, mtime={time.ctime(st.st_mtime)})")
    except Exception as e:
        print(f"[CONFIG] Using: {config_path} (stat failed: {e})")

    print("[CONFIG] Parsed subreddits:", json.dumps(cfg.get("subreddits", [])))
    print("[CONFIG] query:", repr((cfg.get("query","") or "").strip()),
          "| since:", repr(cfg.get("since","") or ""),
          "| until:", repr(cfg.get("until","") or ""))


    subs: List[str] = cfg.get("subreddits", [])
    query: str = (cfg.get("query", "") or "").strip()
    since_iso: str = (cfg.get("since", "") or "").strip()
    until_iso: str = (cfg.get("until", "") or "").strip()

    since = iso_to_epoch(since_iso)
    until = iso_to_epoch(until_iso)

    raw_max = cfg.get("max_posts_per_subreddit", None)
    if raw_max in (None, 0, "0", "", "none", "None"):
        max_posts = None
    else:
        max_posts = to_safe_int(raw_max, None)


    # Comments settings
    fetch_comments = bool(cfg.get("fetch_comments", cfg.get("comments", {}).get("fetch", True)))
    max_comments_per_post = to_safe_int(
        cfg.get("max_comments_per_post", cfg.get("comments", {}).get("max_per_post", 500)),
        500
    )
    comment_depth_cfg = cfg.get("comment_depth", cfg.get("comments", {}).get("depth", "all"))
    depth_limit = normalize_depth(comment_depth_cfg)

    # Output
    out_dir = cfg.get("out_dir", cfg.get("output", {}).get("out_dir", "out"))
    fmt = (cfg.get("format", cfg.get("output", {}).get("format", "csv")) or "csv").lower()
    rotate_every = to_safe_int(
        cfg.get("rotate_every_n_posts", cfg.get("output", {}).get("rotate_every_n_posts", 2000)),
        2000
    )

    # Performance
    sleep_ms = to_safe_int(
        cfg.get("sleep_between_requests_ms", cfg.get("performance", {}).get("sleep_between_requests_ms", 250)),
        250
    )

    # Writers
    posts_writer = RotatingWriter(out_dir, "posts", fmt, rotate_every)
    comments_writer = RotatingWriter(out_dir, "comments", fmt, rotate_every) if fetch_comments and max_comments_per_post != 0 else None

    # Client
    rc = RedditClient()

    # Decide mode:
    # - If query is non-empty -> SEARCH mode (server-side filtering).
    # - If query is empty     -> NEW mode (client-side time filtering using since/until).
    use_search = bool(query)

    # Helpful debug line (safe to keep)
    print(f"MODE: {'search' if use_search else 'new'} | query={repr(query)} | since={since_iso or '∅'} | until={until_iso or '∅'} | subs={subs}")

    # Crawl
    for sub in subs:
        print(f"\n=== Subreddit: r/{sub} ===")
        if use_search:
            it = rc.search_submissions(sub, query=query, since=since, until=until)
        else:
            it = rc.new_submissions(sub)

        count_posts = 0
        for s in tqdm(it, desc=f"posts r/{sub}"):
            p = flatten_submission(s)

            # Local time window guard (always applies when provided)
            if since and p["created_utc"] and p["created_utc"] < since:
                continue
            if until and p["created_utc"] and p["created_utc"] > until:
                continue

            posts_writer.add(p)
            count_posts += 1

            if comments_writer:
                try:
                    comments = crawl_comments_for_submission(
                        submission=s,
                        max_comments=max_comments_per_post,
                        sleep_ms=sleep_ms,
                        depth_limit=depth_limit
                    )
                    for c in comments:
                        comments_writer.add(c)
                except Exception as e:
                    print(f"[warn] comments failed for {p['id']}: {e}", file=sys.stderr)

            if sleep_ms:
                time.sleep(sleep_ms / 1000.0)

            if max_posts and count_posts >= max_posts:
                break

    posts_writer.close()
    if comments_writer:
        comments_writer.close()

    print("\nDone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reddit crawler (posts + comments) with CSV/JSON + depth control.")
    ap.add_argument("--config", default="config.yaml", help="Path to YAML config.")
    args = ap.parse_args()
    run(args)
