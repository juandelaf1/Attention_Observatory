"""Bronze → Silver: normalizes raw per-source parquet into canonical schemas.

Produces:
  data/silver/silver_actors.parquet
  data/silver/silver_posts.parquet
  data/silver/validation_report.json
"""

import os
import json
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

DOMAIN_MAP = {
    "hackernews": "technology", "wikipedia": "science",
    "huggingface": "technology", "bluesky": "technology",
    "mastodon": "technology", "github": "technology",
    "youtube": "technology", "telegram": "technology",
    "reddit": "technology", "openalex": "science",
    "arxiv": "science",
}

PLATFORM_TYPE_MAP = {
    "hackernews": "social_network", "wikipedia": "knowledge_network",
    "huggingface": "corpus_dataset", "bluesky": "social_network",
    "mastodon": "social_network", "github": "collaborative_network",
    "youtube": "social_network", "telegram": "social_network",
    "reddit": "social_network", "openalex": "knowledge_network",
    "arxiv": "knowledge_network",
}

NON_ATTENTION_SOURCES = {"huggingface"}

ACTOR_COLS = [
    "actor_id", "platform", "domain", "platform_type",
    "username", "followers", "following", "content_count",
    "account_created", "description", "is_verified",
    "extraction_ts", "source_version",
]

POST_COLS = [
    "post_id", "actor_id", "platform",
    "timestamp", "title", "content_text", "language",
    "likes", "comments", "shares", "views",
    "sentiment_score",
    "parent_post_id", "reference_id", "reference_type",
    "extraction_ts", "source_version",
]


def _detect_platform(path: str) -> str:
    name = Path(path).stem.lower()
    for p in DOMAIN_MAP:
        if p in name:
            return p
    return "unknown"


def _get_version() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _concat_and_normalize_actors(actor_paths: list[str]) -> tuple[pl.DataFrame, dict]:
    version = _get_version()
    now = datetime.now(timezone.utc).isoformat()
    errors = {"null_counts": {}, "type_errors": 0, "constraint_violations": 0}
    platform_counts = {}
    domain_counts = {}

    rows = []
    for ap in actor_paths:
        if not os.path.exists(ap):
            continue
        platform = _detect_platform(ap)
        df = pl.read_parquet(ap)
        platform_counts[platform] = platform_counts.get(platform, 0) + len(df)
        for row in df.iter_rows(named=True):
            actor_id = row.get("actor_id") or f"{platform}_{row.get('id', 'unknown')}"
            username = str(row.get("username") or row.get("channel_title") or row.get("display_name") or "unknown")
            followers = int(row.get("followers", 0) or 0)
            following = int(row.get("follows", 0) or row.get("following", 0) or 0)
            posts_ct = int(row.get("posts_count", 0) or row.get("total_videos", 0) or row.get("repo_count", 0) or 0)
            desc = str(row.get("description") or row.get("note") or "")
            created = str(row.get("published_at") or row.get("ingested_at", ""))
            is_verified = bool(row.get("type") == "User" if platform == "github" else False)
            extraction = str(row.get("ingested_at", now))
            rows.append({
                "actor_id": actor_id,
                "platform": platform,
                "domain": DOMAIN_MAP.get(platform, "unknown"),
                "platform_type": PLATFORM_TYPE_MAP.get(platform, "unknown"),
                "username": username[:100],
                "followers": max(0, followers),
                "following": max(0, following),
                "content_count": max(0, posts_ct),
                "account_created": created if created != "None" else None,
                "description": desc[:500] if desc else None,
                "is_verified": is_verified,
                "extraction_ts": extraction,
                "source_version": version,
            })
            d = DOMAIN_MAP.get(platform, "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1

    silver = pl.DataFrame(rows, schema=ACTOR_COLS)
    for col in silver.columns:
        n_nulls = silver[col].is_null().sum()
        if n_nulls > 0:
            errors["null_counts"][col] = int(n_nulls)
    return silver, {"platforms": platform_counts, "domains": domain_counts, "errors": errors, "n_actors": len(silver)}


def _concat_and_normalize_posts(post_paths: list[str], actor_ids: set) -> tuple[pl.DataFrame, dict]:
    version = _get_version()
    now = datetime.now(timezone.utc).isoformat()
    errors = {"null_counts": {}, "orphan_posts": 0, "constraint_violations": 0}
    platform_counts = {}
    rows = []
    for pp in post_paths:
        if not os.path.exists(pp):
            continue
        platform = _detect_platform(pp)
        df = pl.read_parquet(pp)
        platform_counts[platform] = platform_counts.get(platform, 0) + len(df)
        for row in df.iter_rows(named=True):
            post_id = str(row.get("post_id", ""))
            actor_id = str(row.get("actor_id", ""))
            if actor_id not in actor_ids:
                errors["orphan_posts"] += 1
                continue
            ts = str(row.get("timestamp", ""))
            title = str(row.get("title", "") or "")[:300]
            content = str(row.get("content_text") or row.get("description") or "")[:2000]
            likes = int(row.get("likes", 0) or 0)
            comments = int(row.get("comments", 0) or 0)
            shares = int(row.get("shares", 0) or 0)
            views = int(row.get("views", 0) or 0)
            sentiment = float(row.get("sentiment_score") or 0.0)
            parent = str(row.get("parent_post_id") or "")
            ref = str(row.get("story_id") or row.get("repo") or row.get("channel_id") or "")
            ref_type = ""
            if row.get("story_id"):
                ref_type = "story"
            elif row.get("repo"):
                ref_type = "repo"
            elif row.get("channel_id"):
                ref_type = "channel"
            extraction = str(row.get("ingested_at", now))
            rows.append({
                "post_id": post_id,
                "actor_id": actor_id,
                "platform": platform,
                "timestamp": ts,
                "title": title if title else None,
                "content_text": content if content else None,
                "language": None,
                "likes": max(0, likes),
                "comments": max(0, comments),
                "shares": max(0, shares),
                "views": max(0, views),
                "sentiment_score": sentiment if sentiment != 0.0 else None,
                "parent_post_id": parent if parent else None,
                "reference_id": ref if ref else None,
                "reference_type": ref_type if ref_type else None,
                "extraction_ts": extraction,
                "source_version": version,
            })
    if not rows:
        return pl.DataFrame(schema=POST_COLS), {"platforms": platform_counts, "n_posts": 0, "errors": errors}
    silver = pl.DataFrame(rows, schema=POST_COLS)
    for col in silver.columns:
        n_nulls = silver[col].is_null().sum()
        if n_nulls > 0 and n_nulls < len(silver):
            errors["null_counts"][col] = int(n_nulls)
    return silver, {"platforms": platform_counts, "errors": errors, "n_posts": len(silver)}


def run_bronze_to_silver(actor_paths: list[str], post_paths: list[str],
                         output_dir: str = "data/silver") -> tuple[str, str, dict]:
    os.makedirs(output_dir, exist_ok=True)
    print(f"[bronze_to_silver] Normalizing {len(actor_paths)} actor files, {len(post_paths)} post files")

    actors, actor_report = _concat_and_normalize_actors(actor_paths)
    actor_ids = set(actors["actor_id"].to_list())
    posts, post_report = _concat_and_normalize_posts(post_paths, actor_ids)

    actors_path = f"{output_dir}/silver_actors.parquet"
    posts_path = f"{output_dir}/silver_posts.parquet"
    actors.write_parquet(actors_path)
    posts.write_parquet(posts_path)

    version = _get_version()
    report = {
        "pipeline": "bronze_to_silver",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_version": version,
        "actors": actor_report,
        "posts": post_report,
        "outputs": {"actors": actors_path, "posts": posts_path},
    }
    report_path = f"{output_dir}/validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[bronze_to_silver] Silver actors -> {actors_path} ({len(actors)} rows)")
    print(f"[bronze_to_silver] Silver posts  -> {posts_path} ({len(posts)} rows)")
    print(f"[bronze_to_silver] Validation   -> {report_path}")
    return actors_path, posts_path, report
