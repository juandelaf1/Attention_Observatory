# REPRODUCIBILITY AUDIT

## Can an independent researcher reproduce all results?

This document evaluates the reproduction chain and identifies gaps.

---

## 1. Pipeline Overview

```
main.py
  ├── _try_hackernews_ingest()    → data/bronze/hn_actors_*.parquet, hn_posts_*.parquet
  ├── _try_wikipedia_ingest()     → data/bronze/wiki_actors_*.parquet, wiki_posts_*.parquet
  ├── _try_huggingface_ingest()   → data/bronze/hf_actors_*.parquet, hf_posts_*.parquet
  ├── _try_bluesky_ingest()       → data/bronze/bluesky_actors_*.parquet, bluesky_posts_*.parquet
  ├── _try_mastodon_ingest()      → data/bronze/mastodon_actors_*.parquet, mastodon_posts_*.parquet
  ├── _try_github_ingest()        → data/bronze/github_actors_*.parquet, github_posts_*.parquet
  ├── _try_reddit_ingest()        → data/bronze/reddit_actors_*.parquet, reddit_posts_*.parquet
  ├── _try_youtube_ingest()       → data/bronze/youtube_actors_*.parquet, youtube_posts_*.parquet
  ├── enrich_posts_with_sentiment() → NLP with distilbert
  ├── run_pipeline()              → bronze_to_silver.py + silver_to_gold.py
  └── run_stats()                 → inequality, concentration, validation
```

---

## 2. Dataset Inventory

| Dataset | Format | Location | Git LFS? | Regenerable? |
|---------|--------|----------|----------|-------------|
| Bronze actors | Parquet | `data/bronze/*_actors_*.parquet` | No (gitignored) | Yes — `python main.py` |
| Bronze posts | Parquet | `data/bronze/*_posts_*.parquet` | No (gitignored) | Yes — `python main.py` |
| Gold metrics | Parquet | `data/gold/fact_metrics.parquet` | No (gitignored) | Yes — `python main.py` |
| Execution snapshots | JSON | `data/executions/*.json` | Yes (tracked) | Yes — `python main.py` |
| Research reports | JSON | `data/reports/research/*.json` | Yes (tracked) | Yes — `python main.py --skip-elt` |
| Quality reports | JSON | `data/reports/*_quality_report.json` | Yes (tracked) | Yes — `python main.py` |

---

## 3. External Dependencies

| Dependency | Version | Purpose | Required? |
|-----------|---------|---------|-----------|
| Python | ≥ 3.11 | Runtime | Yes |
| polars | ≥ 1.0.0 | DataFrame engine | Yes |
| numpy | ≥ 1.26.0 | Numerical computing | Yes |
| scipy | ≥ 1.14.0 | Statistical tests | Yes |
| networkx | ≥ 3.2.0 | Graph analysis | Yes |
| transformers | ≥ 4.44.0 | NLP (distilbert) | Yes |
| torch | ≥ 2.3.0 | ML backend | Yes |
| powerlaw | ≥ 1.5.0 | Power law fitting | Yes |
| streamlit | ≥ 1.35.0 | Dashboard | No (visualization only) |
| plotly | ≥ 5.24.0 | Charts | No (visualization only) |
| matplotlib | ≥ 3.8.0 | Static charts | No (generate_charts.py only) |
| pingouin | ≥ 0.5.0 | Partial correlation | Optional |

---

## 4. Non-Determinism Audit

| Source of randomness | Location | Seeded? | Impact |
|---------------------|----------|---------|--------|
| External API order | `src/ingesta/*.py` | N/A | Non-deterministic (real data changes) |
| `_infer_external_ecosystem` | `silver_to_gold.py` | Uses `np.random.default_rng().binomial(1, prob)` | **NOT SEEDED** — produces different results each run |
| Bootstrap CI | `src/stats/validation.py` | `np.random.default_rng(42)` | Seeded — reproducible |
| Verification sample | `src/ingesta/reddit.py` | `random.sample()` | **NOT SEEDED** — cosmetic only |

**Critical finding**: `_infer_external_ecosystem` uses an unseeded RNG, meaning the `has_external_ecosystem` column differs between runs even with identical input data.

---

## 5. Action Items for Full Reproducibility

| # | Issue | Severity | Fix |
|---|-------|----------|-----|
| 1 | `_infer_external_ecosystem` unseeded | **HIGH** | Replace with deterministic rule or seed `np.random.default_rng(42)` |
| 2 | `random.sample()` in reddit verification | LOW | Add `random.seed(42)` |
| 3 | Pipeline version not recorded in gold | MEDIUM | Add `source_version` to gold (already in silver) |
| 4 | No dataset hash for bronze/gold | MEDIUM | Add SHA-256 hash of input data to execution snapshot |

---

## 6. Reproduction Steps

```powershell
# Fresh reproduction
git clone https://github.com/juandelaf1/attention-observatory.git
cd attention-observatory
conda create -n attention_obs python=3.11
conda activate attention_obs
pip install -r requirements.txt

# Configure API keys (optional — pipeline runs without them)
# $env:GITHUB_TOKEN = '...'
# $env:YOUTUBE_API_KEY = '...'

# Full pipeline (ingestion + transform + stats)
python main.py

# Dashboard
streamlit run app.py
```

---

## 7. Verification Commands

```powershell
# Verify gold structure
python -c "import polars as pl; df=pl.read_parquet('data/gold/fact_metrics.parquet'); print(len(df), df.columns)"

# Verify latest snapshot
python -c "import json; s=json.load(open(sorted(glob.glob('data/executions/*.json'))[-1])); print(s['gini'], s['n_actors'])"

# Run stats only (skip ingestion)
python main.py --skip-elt
```

---

## 8. Gaps

| Gap | Impact | Resolution Timeline |
|-----|--------|-------------------|
| API-dependent data varies daily | Metrics non-comparable across dates | Longitudinal tracking (done) |
| HuggingFace go_emotions dataset version pinned? | Unknown — HF may update dataset | Pin `SetFit/go_emotions` revision |
| YouTube API quota limits | May produce partial results | Document in execution snapshot |
| No DVC for dataset versioning | Cannot roll back to previous data state | Phase 3 (post-validation) |

---

## Conclusion

**Current reproducibility score: 8/10**

An independent researcher can reproduce the pipeline and obtain statistically similar results. Exact numerical replication is limited by:
1. Unseeded `_infer_external_ecosystem` (fix in progress)
2. Real-time API data changes (mitigated by longitudinal snapshots)
3. No dataset hash (mitigation planned)
