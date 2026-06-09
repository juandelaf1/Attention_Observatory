# SILVER SCHEMA SPECIFICATION

## Purpose

Define a canonical, typed, validated intermediate layer between Bronze (raw per-source parquet) and Gold (aggregated feature space). Silver guarantees that all downstream analysis operates on a consistent schema regardless of source heterogeneity.

---

## 1. Silver Actor Schema

```python
actor_id: str           # canonical: "{platform}_{source_specific_id}"
platform: str           # lowercase source name: hackernews, bluesky, ...
domain: str             # taxonomic: technology, science, sports, music, ...
platform_type: str      # functional: social_network, knowledge_network, collaborative_network, corpus_dataset
username: str           # display handle
followers: int          # followers / subscribers / citations
following: int          # accounts followed (0 if not applicable)
content_count: int      # total posts / articles / videos
account_created: str | None   # ISO timestamp
description: str | None       # bio / profile text
is_verified: bool       # platform-verified account
extraction_ts: str      # ISO timestamp of data extraction
source_version: str     # connector version (git commit hash)
```

## 2. Silver Post Schema

```python
post_id: str            # canonical: "{platform}_{source_specific_id}"
actor_id: str           # FK → silver_actors.actor_id
platform: str           # lowercase source name
timestamp: str          # ISO datetime of publication
title: str | None       # post title or first 100 chars of content
content_text: str | None
language: str | None    # detected language code
likes: int
comments: int
shares: int
views: int
sentiment_score: float | None   # [-1, 1] if available
parent_post_id: str | None      # reply/thread parent
reference_id: str | None        # cross-reference (story_id, repo, channel)
reference_type: str | None      # story, repo, channel, article
extraction_ts: str
source_version: str
```

## 3. Type Constraints

| Column | Type | Nullable | Default | Constraint |
|--------|------|----------|---------|-----------|
| actor_id | str | NO | — | Unique, matches `^[a-z]+_[a-zA-Z0-9_-]+$` |
| platform | str | NO | — | Must be in PLATFORM_REGISTRY |
| domain | str | NO | — | Must be in DOMAIN_TAXONOMY |
| platform_type | str | NO | — | Must be in PLATFORM_TYPE_TAXONOMY |
| username | str | NO | "unknown" | — |
| followers | int | NO | 0 | ≥ 0 |
| following | int | NO | 0 | ≥ 0 |
| content_count | int | NO | 0 | ≥ 0 |
| extraction_ts | str | NO | — | ISO 8601 |
| likes | int | NO | 0 | ≥ 0 |
| comments | int | NO | 0 | ≥ 0 |
| shares | int | NO | 0 | ≥ 0 |
| views | int | NO | 0 | ≥ 0 |

## 4. Domain Taxonomy

```
technology        → social_network, collaborative_network, corpus_dataset
science           → knowledge_network
sports            → social_network, database
music             → streaming, database
entertainment     → database
politics          → knowledge_network, database
economy           → database
health            → knowledge_network
education         → knowledge_network
```

## 5. Platform Registry

| platform | domain | platform_type | status |
|----------|--------|--------------|--------|
| hackernews | technology | social_network | active |
| wikipedia | science | knowledge_network | active |
| huggingface | technology | corpus_dataset | active (NLP only) |
| bluesky | technology | social_network | active |
| mastodon | technology | social_network | active |
| github | technology | collaborative_network | active |
| youtube | technology | social_network | active |
| telegram | technology | social_network | pending |
| reddit | technology | social_network | pending |
| openalex | science | knowledge_network | planned |
| arxiv | science | knowledge_network | planned |

## 6. Integrity Rules

1. `actor_id` must be unique across all platforms (prefix guarantees this)
2. `post_id` FK must reference an existing `actor_id`
3. `followers ≥ 0`, `likes ≥ 0`, etc.
4. `timestamp` must be parseable ISO 8601
5. Unknown platforms raise SchemaError

## 7. Validation Output

Each Silver write produces a validation report:

```
{
  "n_actors": int,
  "n_posts": int,
  "platforms": { str: int },
  "domains": { str: int },
  "null_checks": { col: n_nulls },
  "type_errors": int,
  "constraint_violations": int,
  "source_versions": { str: str },
  "extraction_window": [str, str],
}
```
