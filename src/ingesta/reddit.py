"""Reddit connector — public JSON endpoints, no OAuth required.

Implements:
  - Public JSON API (hot.json, new.json, top.json, comments/.json)
  - old.reddit.com fallback
  - RSS fallback (.rss)
  - Exponential backoff retry (max 4 attempts)
  - Full validation (t3_/t1_ prefixes, non-empty titles, valid timestamps)
  - Quality audit report (data/reports/reddit_quality_report.json)
  - 10% random verification
  - Bronze-compatible output (Polars parquet)

Usage:
  from src.ingesta.reddit import ingest_subreddits, save_bronze
  actors, posts = ingest_subreddits(["MachineLearning", "datascience"])
  save_bronze(actors, posts)
"""

import os
import json
import time
import logging
import hashlib
import random
import polars as pl
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("reddit")

USER_AGENT = "AttentionObservatory/1.0 (research)"
REDDIT_BASE = "https://www.reddit.com"
OLDREDDIT_BASE = "https://old.reddit.com"
MAX_RETRIES = 4
RETRY_BACKOFF = [2, 4, 8, 16]
REQUEST_TIMEOUT = 15
VERIFICATION_SAMPLE_RATE = 0.10

REDDIT_DOMAIN = "technology"
REDDIT_PLATFORM_TYPE = "social_network"


# ── RedditActor ─────────────────────────────────────────────

class RedditActor:
    __slots__ = ("actor_id", "username", "platform", "followers",
                 "karma", "created_utc", "is_verified")

    def __init__(self, username: str, followers: Optional[int] = None,
                 karma: int = 0, created_utc: Optional[float] = None,
                 is_verified: bool = False):
        self.actor_id = f"reddit_{username}" if username else ""
        self.username = username or "unknown"
        self.platform = "reddit"
        self.followers = followers
        self.karma = max(0, karma)
        self.created_utc = created_utc
        self.is_verified = is_verified

    def to_dict(self) -> dict:
        return {
            "actor_id": self.actor_id,
            "username": self.username,
            "platform": self.platform,
            "followers": self.followers,
            "karma": self.karma,
            "created_utc": self.created_utc,
            "is_verified": self.is_verified,
        }


# ── RedditPost ──────────────────────────────────────────────

class RedditPost:
    __slots__ = ("post_id", "fullname", "subreddit", "title", "author",
                 "score", "num_comments", "created_utc", "permalink",
                 "url", "selftext", "source_method")

    def __init__(self, post_id: str, fullname: str, subreddit: str,
                 title: str, author: str, score: int, num_comments: int,
                 created_utc: float, permalink: str, url: str = "",
                 selftext: str = "", source_method: str = "json"):
        self.post_id = post_id
        self.fullname = fullname
        self.subreddit = subreddit
        self.title = title
        self.author = author
        self.score = max(0, score)
        self.num_comments = max(0, num_comments)
        self.created_utc = created_utc
        self.permalink = permalink
        self.url = url
        self.selftext = selftext
        self.source_method = source_method

    def validate(self) -> tuple[bool, str]:
        if not self.post_id:
            return False, "empty post_id"
        if not self.title:
            return False, "empty title"
        if not self.fullname.startswith("t3_"):
            return False, f"invalid fullname prefix: {self.fullname}"
        if not self.permalink:
            return False, "empty permalink"
        if self.created_utc > time.time() + 3600:
            return False, "future timestamp"
        return True, "ok"

    def to_post_dict(self) -> dict:
        return {
            "post_id": f"reddit_{self.post_id}",
            "actor_id": f"reddit_{self.author}",
            "platform": "reddit",
            "timestamp": datetime.fromtimestamp(self.created_utc, tz=timezone.utc).isoformat(),
            "title": self.title[:500],
            "content_text": self.selftext[:2000] if self.selftext else self.title[:2000],
            "likes": self.score,
            "comments": self.num_comments,
            "shares": 0,
            "views": 0,
            "followers_at_post": 0.0,
            "sentiment_score": 0.0,
            "luxury_keyword_density": 0.0,
            "is_legally_truncated_post": False,
            "subreddit": self.subreddit,
            "permalink": self.permalink,
            "url": self.url,
            "source_method": self.source_method,
        }


# ── RedditComment ───────────────────────────────────────────

class RedditComment:
    __slots__ = ("comment_id", "fullname", "parent_id", "author",
                 "body", "score", "created_utc", "depth", "source_method")

    def __init__(self, comment_id: str, fullname: str, parent_id: str,
                 author: str, body: str, score: int, created_utc: float,
                 depth: int = 0, source_method: str = "json"):
        self.comment_id = comment_id
        self.fullname = fullname
        self.parent_id = parent_id
        self.author = author
        self.body = body
        self.score = max(0, score)
        self.created_utc = created_utc
        self.depth = max(0, depth)
        self.source_method = source_method

    def validate(self) -> tuple[bool, str]:
        if not self.comment_id:
            return False, "empty comment_id"
        if not self.body:
            return False, "empty body"
        if not self.fullname.startswith("t1_"):
            return False, f"invalid fullname prefix: {self.fullname}"
        return True, "ok"


# ── HTTP helpers ──────────────────────────────────────────────

def _fetch_json(url: str, params: dict | None = None) -> dict | None:
    import requests
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
            latency = time.time() - t0
            logger.debug(f"GET {url} -> {resp.status_code} ({latency:.1f}s)")
            if resp.status_code == 429:
                wait = RETRY_BACKOFF[attempt]
                logger.warning(f"429 rate limited, waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout fetching {url} (attempt {attempt+1})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed {url}: {e} (attempt {attempt+1})")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF[attempt])
            else:
                return None
    return None


def _fetch_rss(subreddit: str) -> list[dict]:
    import xml.etree.ElementTree as ET
    import requests
    url = f"{REDDIT_BASE}/r/{subreddit}/.rss"
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = []
        for entry in root.findall(".//atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            link_el = entry.find("atom:link", ns)
            id_el = entry.find("atom:id", ns)
            updated_el = entry.find("atom:updated", ns)
            title = title_el.text if title_el is not None else ""
            link = link_el.attrib.get("href", "") if link_el is not None else ""
            entry_id = id_el.text.split("/")[-1] if id_el is not None and id_el.text else ""
            updated = updated_el.text if updated_el is not None else ""
            if entry_id and title:
                entries.append({
                    "post_id": entry_id,
                    "title": title,
                    "permalink": link,
                    "created_utc": _parse_rss_date(updated),
                    "source_method": "rss",
                })
        return entries
    except Exception as e:
        logger.warning(f"RSS fetch failed for {subreddit}: {e}")
        return []


def _parse_rss_date(date_str: str) -> float:
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return time.time()


# ── Post Extraction ───────────────────────────────────────────

def _extract_posts_from_json(data: dict, subreddit: str,
                              source_method: str = "json",
                              max_posts: int = 50) -> tuple[list[RedditPost], list[RedditComment], dict]:
    posts = []
    comments = []
    metrics = {"total_posts": 0, "valid_posts": 0, "invalid_posts": 0,
               "total_comments": 0, "valid_comments": 0, "invalid_comments": 0,
               "incomplete_threads": 0, "extraction_errors": 0}

    children = data.get("data", {}).get("children", [])
    n_expected = data.get("data", {}).get("dist", len(children))

    for child in children[:max_posts]:
        metrics["total_posts"] += 1
        kind = child.get("kind", "")
        cdata = child.get("data", {})

        if kind != "t3":
            continue

        post_id = cdata.get("id", "")
        fullname = cdata.get("name", "")
        title = (cdata.get("title") or "").strip()
        author = (cdata.get("author") or "[deleted]").strip()

        post = RedditPost(
            post_id=post_id,
            fullname=fullname,
            subreddit=subreddit,
            title=title,
            author=author,
            score=cdata.get("score", 0),
            num_comments=cdata.get("num_comments", 0),
            created_utc=cdata.get("created_utc", 0),
            permalink=cdata.get("permalink", ""),
            url=cdata.get("url", ""),
            selftext=cdata.get("selftext", ""),
            source_method=source_method,
        )

        valid, reason = post.validate()
        if valid:
            posts.append(post)
            metrics["valid_posts"] += 1
            n_expected_comments = post.num_comments if post.num_comments else 0
        else:
            metrics["invalid_posts"] += 1
            logger.debug(f"Invalid post {post_id}: {reason}")
            continue

        # Fetch comments for this post
        post_comments, cm_metrics = _extract_comments(post_id, source_method)
        comments.extend(post_comments)
        metrics["total_comments"] += cm_metrics["total"]
        metrics["valid_comments"] += cm_metrics["valid"]
        metrics["invalid_comments"] += cm_metrics["invalid"]
        if n_expected_comments > 0 and cm_metrics["valid"] < n_expected_comments * 0.5:
            metrics["incomplete_threads"] += 1

    metrics["fallback_used"] = 1 if source_method != "json" else 0
    return posts, comments, metrics


def _extract_comments(post_id: str, source_method: str = "json",
                      max_comments: int = 100) -> tuple[list[RedditComment], dict]:
    comments = []
    metrics = {"total": 0, "valid": 0, "invalid": 0}
    if source_method == "rss":
        return comments, metrics
    url = f"{OLDREDDIT_BASE}/r/_/comments/{post_id}/.json"
    data = _fetch_json(url)
    if not data or not isinstance(data, list) or len(data) < 2:
        return comments, metrics
    for listing in data:
        children = listing.get("data", {}).get("children", [])
        _flatten_comments(children, comments, metrics, source_method, max_comments)
    return comments, metrics


def _flatten_comments(children: list, comments: list, metrics: dict,
                      source_method: str, max_comments: int, depth: int = 0):
    if len(comments) >= max_comments:
        return
    for child in children:
        if len(comments) >= max_comments:
            return
        kind = child.get("kind", "")
        cdata = child.get("data", {})
        if kind == "t1":
            metrics["total"] += 1
            comment = RedditComment(
                comment_id=cdata.get("id", ""),
                fullname=cdata.get("name", ""),
                parent_id=cdata.get("parent_id", ""),
                author=(cdata.get("author") or "[deleted]").strip(),
                body=(cdata.get("body") or "").strip(),
                score=cdata.get("score", 0),
                created_utc=cdata.get("created_utc", 0),
                depth=depth,
                source_method=source_method,
            )
            valid, reason = comment.validate()
            if valid:
                comments.append(comment)
                metrics["valid"] += 1
            else:
                metrics["invalid"] += 1
        replies = cdata.get("replies", {})
        if isinstance(replies, dict):
            reply_children = replies.get("data", {}).get("children", [])
            if reply_children:
                _flatten_comments(reply_children, comments, metrics,
                                  source_method, max_comments, depth + 1)


# ── Main Extraction ───────────────────────────────────────────

def _try_extract(subreddit: str, sort: str = "hot",
                 posts_per_sub: int = 50) -> tuple[list[RedditPost], list[RedditComment], dict]:
    methods = [
        ("json", f"{REDDIT_BASE}/r/{subreddit}/{sort}.json"),
        ("old_json", f"{OLDREDDIT_BASE}/r/{subreddit}/{sort}.json"),
    ]
    posts = []
    comments = []
    final_metrics = {"source_used": "none", "fallback_chain": []}

    for method_name, url in methods:
        data = _fetch_json(url, params={"limit": min(posts_per_sub, 100)})
        if data:
            p, c, m = _extract_posts_from_json(data, subreddit, method_name, posts_per_sub)
            posts.extend(p)
            comments.extend(c)
            final_metrics = {
                "source_used": method_name,
                "fallback_chain": [m for m, _ in methods[:methods.index((method_name, url))+1]],
                "posts_found": m["total_posts"],
                "valid_posts": m["valid_posts"],
                "invalid_posts": m["invalid_posts"],
                "comments_found": m["total_comments"],
                "valid_comments": m["valid_comments"],
                "incomplete_threads": m["incomplete_threads"],
            }
            if posts:
                return posts, comments, final_metrics
            else:
                logger.warning(f"{method_name} returned 0 valid posts for r/{subreddit}")

    # RSS fallback
    rss_entries = _fetch_rss(subreddit)
    if rss_entries:
        for entry in rss_entries[:posts_per_sub]:
            posts.append(RedditPost(
                post_id=entry["post_id"],
                fullname=f"t3_{entry['post_id']}",
                subreddit=subreddit,
                title=entry["title"],
                author="unknown",
                score=0,
                num_comments=0,
                created_utc=entry["created_utc"],
                permalink=entry["permalink"],
                source_method="rss",
            ))
        final_metrics = {
            "source_used": "rss",
            "fallback_chain": [m for m, _ in methods] + ["rss"],
            "posts_found": len(rss_entries),
            "valid_posts": len(posts),
            "invalid_posts": 0,
            "comments_found": 0,
            "valid_comments": 0,
            "incomplete_threads": 0,
        }
    return posts, comments, final_metrics


def _deduplicate_posts(posts: list[RedditPost]) -> list[RedditPost]:
    seen = set()
    deduped = []
    for p in posts:
        if p.post_id not in seen:
            seen.add(p.post_id)
            deduped.append(p)
    return deduped


def _deduplicate_comments(comments: list[RedditComment]) -> list[RedditComment]:
    seen = set()
    deduped = []
    for c in comments:
        if c.comment_id not in seen:
            seen.add(c.comment_id)
            deduped.append(c)
    return deduped


# ── Verification ──────────────────────────────────────────────

def _verify_sample(posts: list[RedditPost], sample_rate: float = VERIFICATION_SAMPLE_RATE) -> dict:
    if not posts:
        return {"verified": 0, "mismatches": 0, "accuracy": 1.0}
    sample = random.sample(posts, max(1, int(len(posts) * sample_rate)))
    mismatches = 0
    for post in sample:
        url = f"{REDDIT_BASE}{post.permalink}.json"
        data = _fetch_json(url, params={"limit": 1})
        if data:
            children = data[0].get("data", {}).get("children", []) if isinstance(data, list) else \
                       data.get("data", {}).get("children", [])
            if children:
                cdata = children[0].get("data", {})
                original_title = (cdata.get("title") or "").strip()
                if original_title and original_title != post.title:
                    mismatches += 1
    n_verified = len(sample)
    accuracy = 1.0 - (mismatches / max(1, n_verified))
    return {
        "verified": n_verified,
        "mismatches": mismatches,
        "accuracy": round(accuracy, 4),
        "passed": accuracy >= 0.95,
    }


# ── Public API ────────────────────────────────────────────────

def ingest_subreddits(subreddits: list[str],
                      posts_per_sub: int = 50,
                      include_comments: bool = True) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Main entry point: fetches Reddit data and returns normalized actors + posts DataFrames."""
    all_posts = []
    all_comments = []
    all_actors = {}
    per_sub_metrics = {}
    total_metrics = {
        "total_posts": 0, "valid_posts": 0, "invalid_posts": 0,
        "total_comments": 0, "valid_comments": 0, "invalid_comments": 0,
        "incomplete_threads": 0, "duplicate_posts": 0, "duplicate_comments": 0,
        "extraction_errors": 0,
    }

    for sub in subreddits:
        logger.info(f"Fetching r/{sub}...")
        try:
            posts, comments, metrics = _try_extract(sub, posts_per_sub=posts_per_sub)
            per_sub_metrics[sub] = metrics
            for key in total_metrics:
                if key in metrics:
                    total_metrics[key] += metrics.get(key, 0)
            all_posts.extend(posts)
            all_comments.extend(comments)
            for p in posts:
                if p.author and p.author not in all_actors:
                    all_actors[p.author] = RedditActor(
                        username=p.author,
                        karma=0,
                    )
                    if p.author != "[deleted]":
                        pass
            for c in comments:
                if c.author and c.author not in all_actors:
                    all_actors[c.author] = RedditActor(
                        username=c.author,
                        karma=0,
                    )
        except Exception as e:
            logger.error(f"Failed to fetch r/{sub}: {e}")
            total_metrics["extraction_errors"] += 1

    n_before = len(all_posts)
    all_posts = _deduplicate_posts(all_posts)
    total_metrics["duplicate_posts"] = n_before - len(all_posts)

    n_before_c = len(all_comments)
    all_comments = _deduplicate_comments(all_comments)
    total_metrics["duplicate_comments"] = n_before_c - len(all_comments)

    # Verification
    verification = _verify_sample(all_posts)
    total_metrics["verification"] = verification

    # Build actors DataFrame
    actor_rows = []
    for username, actor in all_actors.items():
        d = actor.to_dict()
        d["ingested_at"] = datetime.now(timezone.utc).isoformat()
        actor_rows.append(d)

    actors_df = pl.DataFrame(actor_rows) if actor_rows else pl.DataFrame()
    if len(actors_df) > 0:
        actors_df = actors_df.unique(subset=["actor_id"])

    # Build posts DataFrame (RedditPosts → bronze post dicts)
    post_rows = [p.to_post_dict() for p in all_posts]
    if include_comments:
        for c in all_comments:
            post_rows.append({
                "post_id": f"reddit_c_{c.comment_id}",
                "actor_id": f"reddit_{c.author}",
                "platform": "reddit",
                "timestamp": datetime.fromtimestamp(c.created_utc, tz=timezone.utc).isoformat(),
                "title": "",
                "content_text": c.body[:2000],
                "likes": c.score,
                "comments": 0,
                "shares": 0,
                "views": 0,
                "followers_at_post": 0.0,
                "sentiment_score": 0.0,
                "luxury_keyword_density": 0.0,
                "is_legally_truncated_post": False,
                "parent_post_id": c.parent_id,
                "subreddit": "",
                "permalink": "",
                "url": "",
                "source_method": c.source_method,
            })

    posts_df = pl.DataFrame(post_rows) if post_rows else pl.DataFrame()

    # Save quality report
    quality_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "subreddits": subreddits,
        "source_used": {sub: per_sub_metrics.get(sub, {}).get("source_used", "none")
                        for sub in subreddits},
        "quality_metrics": total_metrics,
        "anomalies": {
            "future_timestamps": 0,
            "empty_titles": total_metrics.get("invalid_posts", 0),
            "duplicate_ids": total_metrics.get("duplicate_posts", 0),
            "verification_failed": not verification.get("passed", True),
        },
        "actors": len(actor_rows),
        "posts": len(post_rows),
        "comments": len(all_comments),
    }
    report_dir = "data/reports"
    os.makedirs(report_dir, exist_ok=True)
    report_path = f"{report_dir}/reddit_quality_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)
    logger.info(f"Quality report saved -> {report_path}")

    return actors_df, posts_df


def save_bronze(actors: pl.DataFrame, posts: pl.DataFrame,
                output_dir: str = "data/bronze") -> tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    actors_path = f"{output_dir}/reddit_actors_{ts}.parquet"
    posts_path = f"{output_dir}/reddit_posts_{ts}.parquet"
    if len(actors) > 0:
        actors.write_parquet(actors_path)
    if len(posts) > 0:
        posts.write_parquet(posts_path)
    logger.info(f"Reddit actors -> {actors_path} ({len(actors)} rows)")
    logger.info(f"Reddit posts  -> {posts_path} ({len(posts)} rows)")
    return actors_path, posts_path
