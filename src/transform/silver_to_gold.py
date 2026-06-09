import polars as pl
import numpy as np
import re

DOMAIN_MAP = {
    "hackernews": "technology",
    "wikipedia": "science",
    "huggingface": "technology",
    "bluesky": "technology",
    "mastodon": "technology",
    "github": "technology",
    "youtube": "technology",
    "telegram": "technology",
    "reddit": "technology",
}

PLATFORM_TYPE_MAP = {
    "hackernews": "social_network",
    "wikipedia": "knowledge_network",
    "huggingface": "corpus_dataset",
    "bluesky": "social_network",
    "mastodon": "social_network",
    "github": "collaborative_network",
    "youtube": "social_network",
    "telegram": "social_network",
    "reddit": "social_network",
}

NON_ATTENTION_SOURCES = {"huggingface"}  # excluded from inequality metrics

AFI_KEYWORDS = [
    "exclusive", "limited", "luxury", "haute", "couture", "gala", "premiere",
    "foundation", "institute", "academy", "award", "ceremony", "editorial",
    "curated", "boutique", "heritage", "legacy", "masterpiece",
    "professor", "phd", "research", "university", "mit", "stanford",
    "harvard", "oxford", "cambridge", "nobel", "science", "study",
    "publication", "journal", "founder", "ceo", "startup", "venture",
    "funding", "series", "ipo", "unicorn", "fellowship", "grant",
    "honor", "exhibition", "collection", "festival", "biennale",
    "laboratory", "observatory", "institute of", "school of",
]


def _ensure_datetime(series: pl.Series) -> pl.Series:
    if series.dtype == pl.Datetime:
        return series
    str_series = series.cast(pl.String)
    cleaned = str_series.str.replace(r"\+00:00$", "+0000").str.replace(r"Z$", "+0000")
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
        try:
            result = cleaned.str.to_datetime(format=fmt, time_zone="UTC", strict=False)
            if result.is_null().sum() < cleaned.len():
                return result
        except Exception:
            continue
    return cleaned.cast(pl.Datetime(time_zone="UTC"), strict=False)


def _normalize_schema(df: pl.DataFrame, required: dict) -> pl.DataFrame:
    for col, default in required.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(default).alias(col))
    return df


def _compute_afi(texts: list[str]) -> list[float]:
    densities = []
    for txt in texts:
        if not txt or len(txt) < 10:
            densities.append(0.0)
        else:
            lower = txt.lower()
            count = sum(1 for kw in AFI_KEYWORDS if kw in lower)
            densities.append(min(1.0, count / max(1, len(txt)) * 100))
    return densities


def _infer_external_ecosystem(followers: np.ndarray, platform: list[str]) -> list[bool]:
    result = []
    for f, p in zip(followers, platform):
        threshold = 500000 if p == "reddit" else 100000
        prob = 1 / (1 + np.exp(-0.8 * (np.log10(max(f, 1)) - np.log10(max(threshold, 1)))))
        result.append(bool(np.random.default_rng().binomial(1, prob)))
    return result


def _detect_truncation(posts: pl.DataFrame) -> pl.DataFrame:
    truncated = (
        posts
        .sort(["actor_id", "timestamp"])
        .group_by("actor_id")
        .agg([
            pl.col("ppi").alias("ppi_series"),
            pl.col("engagement_rate").alias("er_series"),
            pl.len().alias("n_posts"),
        ])
        .with_columns([
            pl.col("ppi_series").list.eval(
                pl.element().diff().fill_null(0)
            ).alias("ppi_deltas"),
        ])
    )

    def _is_truncated(ppi_series, er_series, n):
        if n < 5:
            return False
        ppi_arr = np.array(ppi_series)
        er_arr = np.array(er_series)
        if len(ppi_arr) < 3:
            return False
        second_half = ppi_arr[len(ppi_arr)//2:]
        first_half = ppi_arr[:len(ppi_arr)//2]
        if len(second_half) < 2:
            return False
        if np.mean(second_half) < 0.1 and np.mean(first_half) > 0.5:
            return True
        if np.std(ppi_arr[-3:]) < 0.01 and np.mean(ppi_arr[-3:]) < 0.05 and np.mean(er_arr[-3:]) < np.mean(er_arr[:3]) * 0.1:
            return True
        return False

    trunc_map = {}
    for row in truncated.iter_rows(named=True):
        trunc_map[row["actor_id"]] = _is_truncated(
            row["ppi_series"], row["er_series"], row["n_posts"]
        )

    return posts.with_columns(
        pl.col("actor_id").replace_strict(
            trunc_map, default=False
        ).alias("_is_legally_truncated")
    )


def _compute_prestige(posts: pl.DataFrame) -> pl.DataFrame:
    prestige = (
        posts
        .group_by("actor_id")
        .agg([
            pl.col("ppi").mean().alias("avg_ppi"),
            pl.col("engagement_rate").mean().alias("avg_er"),
            pl.col("luxury_keyword_density").mean().alias("avg_afi"),
            pl.len().alias("post_count"),
        ])
        .with_columns([
            pl.when(
                (pl.col("avg_ppi") < pl.col("avg_ppi").median()) &
                (pl.col("avg_afi") > pl.col("avg_afi").median())
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias("has_prestige_trajectory"),
        ])
    )
    return prestige.select(["actor_id", "has_prestige_trajectory"])


def _concat_dataframes(dfs: list[pl.DataFrame]) -> pl.DataFrame:
    valid = [df for df in dfs if df.height > 0]
    if not valid:
        return pl.DataFrame()
    return pl.concat(valid, how="diagonal")


def run_pipeline(
    actors_paths: list[str],
    posts_paths: list[str],
    output_dir: str = "data/gold",
) -> str:
    import os
    os.makedirs(output_dir, exist_ok=True)

    actor_dfs = []
    post_dfs = []
    for ap in actors_paths:
        if os.path.exists(ap):
            actor_dfs.append(pl.read_parquet(ap))
    for pp in posts_paths:
        if os.path.exists(pp):
            post_dfs.append(pl.read_parquet(pp))

    if not actor_dfs or not post_dfs:
        print("[transform] No data found.")
        raise FileNotFoundError("No bronze parquet files found to transform.")

    actors = _concat_dataframes(actor_dfs).unique(subset=["actor_id"])
    posts = _concat_dataframes(post_dfs)

    print(f"[transform] Loaded {len(actors)} actors, {len(posts)} posts total")

    posts = _normalize_schema(posts, {
        "likes": 0, "comments": 0, "shares": 0, "views": 0,
        "followers_at_post": 0.0, "sentiment_score": 0.0,
        "luxury_keyword_density": 0.0, "content_text": "",
    })

    if posts["luxury_keyword_density"].is_null().sum() > 0 or posts["luxury_keyword_density"].eq(0).all():
        texts = posts["content_text"].to_list()
        densities = _compute_afi(texts)
        posts = posts.with_columns(pl.Series("luxury_keyword_density", densities))

    posts = posts.with_columns(_ensure_datetime(posts["timestamp"]))

    posts = posts.with_columns([
        pl.col("likes").fill_null(0).cast(pl.Int64),
        pl.col("comments").fill_null(0).cast(pl.Int64),
        pl.col("shares").fill_null(0).cast(pl.Int64),
        pl.col("views").fill_null(0).cast(pl.Int64),
        pl.col("followers_at_post").fill_null(0).cast(pl.Float64),
        pl.col("sentiment_score").fill_null(0.0).cast(pl.Float64),
        pl.col("luxury_keyword_density").fill_null(0.0).cast(pl.Float64),
    ])

    posts = posts.with_columns(
        (pl.col("likes") + pl.col("comments") + pl.col("shares")).alias("interactions_raw")
    )

    if "followers" in actors.columns:
        follower_map = actors.select(["actor_id", "followers"]).to_dict(as_series=False)
        fmap = dict(zip(follower_map["actor_id"], follower_map["followers"]))
        posts = posts.with_columns(
            pl.col("actor_id").replace_strict(fmap, default=0).alias("_actor_followers")
        )
        posts = posts.with_columns(
            pl.when(pl.col("followers_at_post") == 0)
            .then(pl.col("_actor_followers"))
            .otherwise(pl.col("followers_at_post"))
            .alias("followers_at_post")
        )

    posts = posts.with_columns(
        pl.when(pl.col("followers_at_post") == 0)
        .then(pl.lit(1.0))
        .otherwise(pl.col("followers_at_post"))
        .alias("followers_at_post")
    )

    posts = posts.sort(["actor_id", "timestamp"])
    posts = posts.with_columns([
        (pl.col("timestamp").diff().dt.total_hours()
         .over("actor_id").fill_null(168.0))
        .alias("posting_interval_hours")
    ])
    posts = posts.with_columns(
        (1.0 / (pl.col("posting_interval_hours").log1p() + 0.01)).alias("ppi")
    )
    posts = posts.with_columns(
        ((pl.col("interactions_raw") / pl.col("followers_at_post")) * 100).alias("engagement_rate")
    )

    posts = _detect_truncation(posts)

    agg = posts.group_by("actor_id").agg([
        pl.col("engagement_rate").mean().alias("er_mean"),
        pl.col("engagement_rate").std().fill_null(0.0).alias("er_std"),
        pl.col("ppi").mean().alias("ppi_mean"),
        pl.col("ppi").std().fill_null(0.0).alias("ppi_std"),
        pl.col("interactions_raw").sum().alias("total_interactions"),
        pl.col("followers_at_post").max().alias("max_followers"),
        pl.col("_is_legally_truncated").max().alias("is_legally_truncated"),
        pl.len().alias("post_count"),
    ])

    sentiment_agg = posts.group_by("actor_id").agg([
        pl.col("sentiment_score").mean().alias("sentiment_avg"),
        pl.col("sentiment_score").std().fill_null(0.0).alias("sentiment_variance"),
    ])

    afi_agg = posts.group_by("actor_id").agg([
        pl.col("luxury_keyword_density").mean().alias("afi_mean"),
        pl.col("luxury_keyword_density").max().alias("afi_max"),
    ])

    prestige = _compute_prestige(posts)

    actors = _normalize_schema(actors, {
        "has_external_ecosystem": False,
        "is_legally_truncated": False,
        "has_prestige_trajectory": False,
        "platform": "unknown",
    })

    if actors["has_external_ecosystem"].is_null().all() or actors["has_external_ecosystem"].eq(False).all():
        followers = actors["followers"].fill_null(0).to_numpy()
        platform_col = actors["platform"] if "platform" in actors.columns else pl.Series([""] * len(actors))
        platform = platform_col.to_list()
        actors = actors.with_columns(
            pl.Series("has_external_ecosystem", _infer_external_ecosystem(followers, platform))
        )

    gold = (
        actors
        .join(agg, on="actor_id", how="left")
        .join(sentiment_agg, on="actor_id", how="left")
        .join(afi_agg, on="actor_id", how="left")
        .join(prestige, on="actor_id", how="left")
        .with_columns([
            pl.col("er_mean").fill_null(0.0),
            pl.col("ppi_mean").fill_null(0.0),
            pl.col("sentiment_avg").fill_null(0.0),
            pl.col("sentiment_variance").fill_null(0.0),
            pl.col("afi_mean").fill_null(0.0),
            pl.col("afi_max").fill_null(0.0),
            pl.col("total_interactions").fill_null(0).cast(pl.Int64),
            pl.col("post_count").fill_null(0).cast(pl.Int64),
            pl.col("is_legally_truncated").fill_null(False),
            pl.col("has_external_ecosystem").fill_null(False),
            pl.col("has_prestige_trajectory").fill_null(False),
        ])
        .with_columns([
            pl.col("platform").replace_strict(DOMAIN_MAP, default="unknown").alias("domain"),
            pl.col("platform").replace_strict(PLATFORM_TYPE_MAP, default="unknown").alias("platform_type"),
            pl.when(pl.col("platform").is_in(NON_ATTENTION_SOURCES)).then(pl.lit(False)).otherwise(pl.lit(True)).alias("is_attention_source"),
            pl.when(pl.col("platform") == "huggingface").then(pl.lit(True)).otherwise(pl.lit(False)).alias("is_huggingface"),
        ])
    )

    gold = gold.with_columns(
        pl.when(
            pl.col("has_prestige_trajectory") & (pl.col("ppi_mean") < pl.col("ppi_mean").median())
        ).then(pl.lit(True))
        .otherwise(pl.lit(False))
        .alias("prestige_drift_detected")
    )

    gold_path = f"{output_dir}/fact_metrics.parquet"
    gold.write_parquet(gold_path)
    print(f"[transform] Gold feature space -> {gold_path} ({len(gold)} rows)")
    return gold_path
