"""Tests for the Reddit connector."""

import sys, os, json, time, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl
from src.ingesta.reddit import (
    RedditPost, RedditComment, RedditActor,
    _deduplicate_posts, _deduplicate_comments,
    ingest_subreddits, save_bronze,
)


# ── Post Parsing ─────────────────────────────────────────────

class TestPostParsing:
    def test_valid_post(self):
        p = RedditPost(
            post_id="abc123", fullname="t3_abc123",
            subreddit="test", title="Hello World",
            author="user1", score=42, num_comments=5,
            created_utc=time.time() - 3600, permalink="/r/test/comments/abc123/",
        )
        valid, reason = p.validate()
        assert valid, f"Expected valid, got: {reason}"
        assert p.to_post_dict()["post_id"] == "reddit_abc123"

    def test_invalid_fullname(self):
        p = RedditPost(
            post_id="abc123", fullname="t2_abc123",
            subreddit="test", title="Hello",
            author="user1", score=0, num_comments=0,
            created_utc=time.time(), permalink="/r/test/",
        )
        valid, reason = p.validate()
        assert not valid
        assert "t2_" in reason

    def test_empty_title(self):
        p = RedditPost(
            post_id="abc123", fullname="t3_abc123",
            subreddit="test", title="",
            author="user1", score=0, num_comments=0,
            created_utc=time.time(), permalink="/r/test/",
        )
        valid, reason = p.validate()
        assert not valid
        assert "empty title" in reason

    def test_future_timestamp(self):
        p = RedditPost(
            post_id="abc123", fullname="t3_abc123",
            subreddit="test", title="Hello",
            author="user1", score=0, num_comments=0,
            created_utc=time.time() + 99999, permalink="/r/test/",
        )
        valid, reason = p.validate()
        assert not valid
        assert "future" in reason


# ── Comment Parsing ──────────────────────────────────────────

class TestCommentParsing:
    def test_valid_comment(self):
        c = RedditComment(
            comment_id="def456", fullname="t1_def456",
            parent_id="t3_abc123", author="user2",
            body="Great post!", score=10,
            created_utc=time.time() - 1800,
        )
        valid, reason = c.validate()
        assert valid, f"Expected valid, got: {reason}"

    def test_invalid_comment_fullname(self):
        c = RedditComment(
            comment_id="def456", fullname="t2_def456",
            parent_id="t3_abc123", author="user2",
            body="Great!", score=0,
            created_utc=time.time(),
        )
        valid, reason = c.validate()
        assert not valid
        assert "t1_" in reason

    def test_empty_body(self):
        c = RedditComment(
            comment_id="def456", fullname="t1_def456",
            parent_id="t3_abc123", author="user2",
            body="", score=0,
            created_utc=time.time(),
        )
        valid, reason = c.validate()
        assert not valid
        assert "empty body" in reason


# ── Deduplication ────────────────────────────────────────────

class TestDeduplication:
    def test_deduplicate_posts(self):
        posts = [
            RedditPost("a", "t3_a", "t", "A", "u", 0, 0, 100, "/a/"),
            RedditPost("b", "t3_b", "t", "B", "u", 0, 0, 200, "/b/"),
            RedditPost("a", "t3_a", "t", "A dup", "u2", 0, 0, 300, "/a/"),
        ]
        deduped = _deduplicate_posts(posts)
        assert len(deduped) == 2
        assert deduped[0].post_id == "a"
        assert deduped[1].post_id == "b"

    def test_deduplicate_comments(self):
        comments = [
            RedditComment("c1", "t1_c1", "t3_a", "u", "hi", 0, 100),
            RedditComment("c2", "t1_c2", "t3_a", "u", "hey", 0, 200),
            RedditComment("c1", "t1_c1", "t3_a", "u2", "hi dup", 0, 300),
        ]
        deduped = _deduplicate_comments(comments)
        assert len(deduped) == 2


# ── Bronze Export ─────────────────────────────────────────────

class TestBronzeExport:
    def test_save_bronze(self):
        actors = pl.DataFrame({
            "actor_id": ["reddit_user1"],
            "username": ["user1"],
            "platform": ["reddit"],
            "followers": [None],
            "karma": [100],
            "created_utc": [time.time()],
            "is_verified": [False],
            "ingested_at": ["2024-01-01T00:00:00"],
        })
        posts = pl.DataFrame({
            "post_id": ["reddit_abc123"],
            "actor_id": ["reddit_user1"],
            "platform": ["reddit"],
            "timestamp": ["2024-01-01T00:00:00"],
            "title": ["Test"],
            "content_text": ["Test content"],
            "likes": [42],
            "comments": [5],
            "shares": [0],
            "views": [0],
            "followers_at_post": [0.0],
            "sentiment_score": [0.0],
            "luxury_keyword_density": [0.0],
            "is_legally_truncated_post": [False],
            "subreddit": ["test"],
            "permalink": ["/r/test/"],
            "url": [""],
            "source_method": ["json"],
        })
        with tempfile.TemporaryDirectory() as tmp:
            a_path, p_path = save_bronze(actors, posts, tmp)
            assert os.path.exists(a_path)
            assert os.path.exists(p_path)
            df_a = pl.read_parquet(a_path)
            df_p = pl.read_parquet(p_path)
            assert len(df_a) == 1
            assert len(df_p) == 1


# ── Retry / Timeout ──────────────────────────────────────────

class TestRetry:
    def test_429_retry_logic(self, monkeypatch):
        """Verify the fetch function respects retry limits."""
        from src.ingesta.reddit import _fetch_json, MAX_RETRIES
        import requests
        call_count = 0

        def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            resp = requests.Response()
            resp.status_code = 429
            resp.url = args[0]
            raise requests.exceptions.HTTPError(response=resp)

        monkeypatch.setattr(requests, "get", mock_get)
        result = _fetch_json("https://www.reddit.com/r/test/hot.json")
        assert result is None
        assert call_count == MAX_RETRIES


# ── Quality Report ───────────────────────────────────────────

class TestQualityReport:
    def test_report_generated(self):
        """ingest_subreddits should generate a quality report."""
        # Use a real but small subreddit
        actors, posts = ingest_subreddits(["test"], posts_per_sub=3)
        report_path = "data/reports/reddit_quality_report.json"
        assert os.path.exists(report_path), "Quality report not generated"
        with open(report_path) as f:
            report = json.load(f)
        assert "quality_metrics" in report
        assert "subreddits" in report
        assert "test" in report["subreddits"]
