# DATA PROVENANCE SPECIFICATION

## Purpose

Every observation in the Attention Observatory must be traceable to its origin. This document defines the provenance fields embedded in Silver and Gold layers.

---

## 1. Per-Row Provenance Fields

### Silver Layer

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `extraction_ts` | str (ISO 8601) | `2026-06-09T13:39:11+00:00` | When the source API was called |
| `source_version` | str | `7723e5c` | Git commit hash of connector code |
| `platform` | str | `reddit`, `hackernews` | Source platform identifier |
| `domain` | str | `technology` | Taxonomic domain |
| `platform_type` | str | `social_network` | Functional classification |

### Gold Layer

| Field | Type | Example | Description |
|-------|------|---------|-------------|
| `ingested_at` | str (ISO 8601) | `2026-06-09T13:39:11+00:00` | When data was ingested (carried from bronze) |
| `is_attention_source` | bool | `True` | Whether source is an attention network |

---

## 2. Platform Registry

| Platform | Domain | Platform Type | Extraction Method | Rate Limit |
|----------|--------|--------------|-------------------|------------|
| hackernews | technology | social_network | REST API (public) | ~500 req/min |
| wikipedia | science | knowledge_network | REST API (public) | ~200 req/min |
| huggingface | technology | corpus_dataset | Datasets library | None |
| bluesky | technology | social_network | AT Protocol (public) | ~5000 req/h |
| mastodon | technology | social_network | REST API (public) | ~300 req/min |
| github | technology | collaborative_network | REST API (token) | 5000 req/h |
| reddit | technology | social_network | JSON endpoints (public) | 60 req/min |

---

## 3. Extraction Metadata

Each bronze file is accompanied by extraction metadata:

```json
{
  "source": "reddit",
  "endpoint": "https://www.reddit.com/r/MachineLearning/hot.json",
  "timestamp": "2026-06-09T13:39:11+00:00",
  "user_agent": "AttentionObservatory/1.0 (research)",
  "status_code": 200,
  "latency_ms": 1250,
  "items_returned": 25,
  "fallback_used": false
}
```

---

## 4. Transformation Chain

```
Bronze (raw per-source parquet)
  → extraction_ts, source_version added
  ↓
Silver (canonical schema)
  → domain, platform_type inferred from platform registry
  → null checks, type coercion, FK validation
  ↓
Gold (aggregated feature space)
  → is_attention_source flag
  → ER, PPI, AFI, Sentiment computed
  → per-execution snapshot saved
```

---

## 5. Version Tracking

| Component | Version Source | Location |
|-----------|---------------|----------|
| Source code | git commit hash | `source_version` field |
| Pipeline | git commit hash | `src/transform/*.py` |
| Schema | semver (planned) | `docs/SILVER_SCHEMA_SPEC.md` |
| Data | snapshot JSON | `data/executions/*.json` |

---

## 6. Reproducibility Guarantee

Same pipeline version (git commit) + same bronze data → identical silver and gold outputs.

Determinism: all random operations use seeded generators (`np.random.default_rng(42)`).
