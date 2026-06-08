import argparse
import sys
import os
import polars as pl
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from src.transform.silver_to_gold import run_pipeline
try:
    from src.nlp.sentiment import enrich_posts_with_sentiment
except ImportError:
    def enrich_posts_with_sentiment(p, **kw):
        print("[nlp] transformers not available; using embedded sentiment from bronze")
        return p
from src.stats.inequality import (
    compute_inequality,
    compute_anomalies,
    compute_breakdown,
    summary_text,
)


def _try_youtube_ingest(bronze_dir: str) -> tuple[list[str], list[str]]:
    actors_paths, posts_paths = [], []
    try:
        from src.ingesta.youtube import ingest_channel_feed, save_bronze

        query = os.environ.get("YOUTUBE_SEARCH_QUERY", "digital creator")
        max_channels = int(os.environ.get("YOUTUBE_MAX_CHANNELS", "5"))
        max_vids = int(os.environ.get("YOUTUBE_MAX_VIDEOS", "20"))

        print(f"[main] YouTube ingest: query='{query}', channels={max_channels}")
        actors, posts = ingest_channel_feed(query, max_channels=max_channels, max_videos_per_channel=max_vids)

        if len(actors) > 0:
            a_path, p_path = save_bronze(actors, posts, bronze_dir)
            actors_paths.append(a_path)
            posts_paths.append(p_path)
            print(f"[main] YouTube: {len(actors)} actors, {len(posts)} posts")
        else:
            print("[main] YouTube: no data returned")
    except ValueError as e:
        print(f"[main] YouTube skipped: {e}")
    except Exception as e:
        print(f"[main] YouTube error: {e}")
    return actors_paths, posts_paths


def _try_reddit_ingest(bronze_dir: str) -> tuple[list[str], list[str]]:
    actors_paths, posts_paths = [], []
    try:
        from src.ingesta.reddit import ingest_subreddits, save_bronze

        subs = os.environ.get("REDDIT_SUBREDDITS", "MachineLearning,datascience,artificial")
        sub_list = [s.strip() for s in subs.split(",") if s.strip()]
        posts_per = int(os.environ.get("REDDIT_POSTS_PER_SUB", "50"))

        print(f"[main] Reddit ingest: subreddits={sub_list}")
        actors, posts = ingest_subreddits(sub_list, posts_per_sub=posts_per)

        if len(actors) > 0:
            a_path, p_path = save_bronze(actors, posts, bronze_dir)
            actors_paths.append(a_path)
            posts_paths.append(p_path)
            print(f"[main] Reddit: {len(actors)} actors, {len(posts)} posts")
        else:
            print("[main] Reddit: no data returned")
    except ValueError as e:
        print(f"[main] Reddit skipped: {e}")
    except Exception as e:
        print(f"[main] Reddit error: {e}")
    return actors_paths, posts_paths


def _try_huggingface_ingest(bronze_dir: str) -> tuple[list[str], list[str]]:
    actors_paths, posts_paths = [], []
    try:
        from src.ingesta.huggingface import ingest_multiple_datasets, save_bronze

        dataset_name = os.environ.get("HF_DATASET", "SetFit/go_emotions")
        max_rows = int(os.environ.get("HF_MAX_ROWS", "3000"))
        cfg = [{"name": dataset_name, "split": "train", "max_rows": max_rows, "text_col": "text", "platform": "huggingface"}]

        print(f"[main] HuggingFace ingest: {dataset_name} ({max_rows} rows)")
        actors, posts = ingest_multiple_datasets(cfg)
        if len(actors) > 0:
            a_path, p_path = save_bronze(actors, posts, bronze_dir)
            actors_paths.append(a_path)
            posts_paths.append(p_path)
            print(f"[main] HuggingFace: {len(actors)} actors, {len(posts)} posts")
    except Exception as e:
        print(f"[main] HuggingFace error: {e}")
    return actors_paths, posts_paths


def _try_bluesky_ingest(bronze_dir: str) -> tuple[list[str], list[str]]:
    actors_paths, posts_paths = [], []
    try:
        from src.ingesta.bluesky import ingest_search, save_bronze

        topics_str = os.environ.get("BLUESKY_TOPICS",
            "artificial intelligence,technology,startup,research,climate,science,philosophy,data")
        topics = [t.strip() for t in topics_str.split(",") if t.strip()]
        per_topic = int(os.environ.get("BLUESKY_PER_TOPIC", "30"))

        print(f"[main] Bluesky ingest: {len(topics)} topics, {per_topic} each")
        actors, posts = ingest_search(topics=topics, posts_per_topic=per_topic)
        if len(actors) > 0:
            a_path, p_path = save_bronze(actors, posts, bronze_dir)
            actors_paths.append(a_path)
            posts_paths.append(p_path)
            print(f"[main] Bluesky: {len(actors)} actors, {len(posts)} posts")
    except Exception as e:
        print(f"[main] Bluesky error: {e}")
    return actors_paths, posts_paths


def _try_mastodon_ingest(bronze_dir: str) -> tuple[list[str], list[str]]:
    actors_paths, posts_paths = [], []
    try:
        from src.ingesta.mastodon import ingest_public_timelines, save_bronze

        instances_str = os.environ.get("MASTODON_INSTANCES", "mastodon.world,techhub.social,fosstodon.org")
        instances = [s.strip() for s in instances_str.split(",") if s.strip()]
        per_instance = int(os.environ.get("MASTODON_PER_INSTANCE", "40"))

        print(f"[main] Mastodon ingest: {len(instances)} instances, {per_instance} each")
        actors, posts = ingest_public_timelines(instances=instances, posts_per_instance=per_instance)
        if len(actors) > 0:
            a_path, p_path = save_bronze(actors, posts, bronze_dir)
            actors_paths.append(a_path)
            posts_paths.append(p_path)
            print(f"[main] Mastodon: {len(actors)} actors, {len(posts)} posts")
    except Exception as e:
        print(f"[main] Mastodon error: {e}")
    return actors_paths, posts_paths


def _try_github_ingest(bronze_dir: str) -> tuple[list[str], list[str]]:
    actors_paths, posts_paths = [], []
    try:
        from src.ingesta.github import ingest_github, save_bronze

        queries_str = os.environ.get("GITHUB_QUERIES", "stars:>10000,topic:machine-learning,topic:artificial-intelligence,topic:data-science,topic:blockchain")
        queries = [q.strip() for q in queries_str.split(",") if q.strip()]
        repos_per = int(os.environ.get("GITHUB_REPOS_PER_QUERY", "3"))
        commits_per = int(os.environ.get("GITHUB_COMMITS_PER_REPO", "5"))
        issues_per = int(os.environ.get("GITHUB_ISSUES_PER_REPO", "3"))

        print(f"[main] GitHub ingest: {len(queries)} queries, {repos_per} repos, {commits_per} commits, {issues_per} issues")
        actors, posts = ingest_github(queries=queries, repos_per_query=repos_per, commits_per_repo=commits_per, issues_per_repo=issues_per)
        if len(actors) > 0:
            a_path, p_path = save_bronze(actors, posts, bronze_dir)
            actors_paths.append(a_path)
            posts_paths.append(p_path)
            print(f"[main] GitHub: {len(actors)} actors, {len(posts)} posts")
    except Exception as e:
        print(f"[main] GitHub error: {e}")
    return actors_paths, posts_paths


def _try_telegram_ingest(bronze_dir: str) -> tuple[list[str], list[str]]:
    actors_paths, posts_paths = [], []
    try:
        from src.ingesta.telegram import ingest_telegram_messages, save_bronze

        channels_str = os.environ.get("TELEGRAM_CHANNELS", "techcrunch,BBCNews,nytimes,NatureNews")
        channels = [c.strip() for c in channels_str.split(",") if c.strip()]
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        per_channel = int(os.environ.get("TELEGRAM_PER_CHANNEL", "50"))

        if token:
            print(f"[main] Telegram ingest: {len(channels)} channels, {per_channel} each")
            actors, posts = ingest_telegram_messages(channels, token, max_messages_per_channel=per_channel)
            if len(actors) > 0:
                a_path, p_path = save_bronze(actors, posts, bronze_dir)
                actors_paths.append(a_path)
                posts_paths.append(p_path)
                print(f"[main] Telegram: {len(actors)} actors, {len(posts)} posts")
        else:
            print("[main] Telegram skipped: TELEGRAM_BOT_TOKEN not set")
    except Exception as e:
        print(f"[main] Telegram error: {e}")
    return actors_paths, posts_paths


def _try_hackernews_ingest(bronze_dir: str) -> tuple[list[str], list[str]]:
    from src.ingesta.hackernews import ingest_top_stories, save_bronze
    n_stories = int(os.environ.get("HN_TOP_STORIES", "30"))
    n_comments = int(os.environ.get("HN_COMMENTS_PER_STORY", "10"))
    print(f"[main] HackerNews ingest: {n_stories} stories, {n_comments} comments each")
    actors, posts = ingest_top_stories(n_stories, n_comments)
    if len(actors) > 0:
        a_path, p_path = save_bronze(actors, posts, bronze_dir)
        print(f"[main] HackerNews: {len(actors)} actors, {len(posts)} posts")
        return [a_path], [p_path]
    return [], []


def _try_wikipedia_ingest(bronze_dir: str) -> tuple[list[str], list[str]]:
    from src.ingesta.wikipedia import ingest_wikipedia_articles, save_bronze
    topics = os.environ.get("WIKI_TOPICS", "Artificial intelligence,Machine learning,Data science,Social media,Climate change,Quantum computing,Neuroscience,Space exploration")
    topic_list = [t.strip() for t in topics.split(",") if t.strip()]
    per_query = int(os.environ.get("WIKI_PER_QUERY", "5"))
    revisions = int(os.environ.get("WIKI_REVISIONS", "10"))
    print(f"[main] Wikipedia ingest: {len(topic_list)} topics, {per_query} articles each, {revisions} revisions")
    actors, posts = ingest_wikipedia_articles(topic_list, articles_per_query=per_query, max_revisions=revisions)
    if len(actors) > 0:
        a_path, p_path = save_bronze(actors, posts, bronze_dir)
        print(f"[main] Wikipedia: {len(actors)} editors, {len(posts)} revisions")
        return [a_path], [p_path]
    return [], []


def _try_simulator_fallback(bronze_dir: str) -> tuple[list[str], list[str]]:
    from src.ingesta.simulator import generate_bronze
    print("[main] Using simulated data (fallback). Set YOUTUBE_API_KEY or REDDIT_CLIENT_ID for real data.")
    a_path, p_path = generate_bronze(bronze_dir)
    return [a_path], [p_path]


def _find_bronze_files(bronze_dir: str) -> tuple[list[str], list[str]]:
    actors = sorted(Path(bronze_dir).glob("*actors*.parquet"))
    posts = sorted(Path(bronze_dir).glob("*posts*.parquet"))
    actors_paths = [str(f) for f in actors]
    posts_paths = [str(f) for f in posts]
    return actors_paths, posts_paths


def run_elt(force_simulator: bool = False):
    print("=" * 60)
    print("ATTENTION OBSERVATORY — ELT PIPELINE")
    print("=" * 60)

    bronze_dir = "data/bronze"
    silver_dir = "data/silver"
    gold_dir = "data/gold"

    for d in [bronze_dir, silver_dir, gold_dir]:
        os.makedirs(d, exist_ok=True)

    if force_simulator:
        actors_paths, posts_paths = _try_simulator_fallback(bronze_dir)
    else:
        actors_paths, posts_paths = [], []

        hn_actors, hn_posts = _try_hackernews_ingest(bronze_dir)
        actors_paths.extend(hn_actors)
        posts_paths.extend(hn_posts)

        wiki_actors, wiki_posts = _try_wikipedia_ingest(bronze_dir)
        actors_paths.extend(wiki_actors)
        posts_paths.extend(wiki_posts)

        hf_actors, hf_posts = _try_huggingface_ingest(bronze_dir)
        actors_paths.extend(hf_actors)
        posts_paths.extend(hf_posts)

        bsky_actors, bsky_posts = _try_bluesky_ingest(bronze_dir)
        actors_paths.extend(bsky_actors)
        posts_paths.extend(bsky_posts)

        masto_actors, masto_posts = _try_mastodon_ingest(bronze_dir)
        actors_paths.extend(masto_actors)
        posts_paths.extend(masto_posts)

        gh_actors, gh_posts = _try_github_ingest(bronze_dir)
        actors_paths.extend(gh_actors)
        posts_paths.extend(gh_posts)

        tg_actors, tg_posts = _try_telegram_ingest(bronze_dir)
        actors_paths.extend(tg_actors)
        posts_paths.extend(tg_posts)

        yt_actors, yt_posts = _try_youtube_ingest(bronze_dir)
        actors_paths.extend(yt_actors)
        posts_paths.extend(yt_posts)

        red_actors, red_posts = _try_reddit_ingest(bronze_dir)
        actors_paths.extend(red_actors)
        posts_paths.extend(red_posts)

        if not actors_paths:
            actors_paths, posts_paths = _try_simulator_fallback(bronze_dir)

    for pp in posts_paths:
        try:
            enrich_posts_with_sentiment(pp)
        except Exception as e:
            print(f"[main] NLP enrichment failed for {pp}: {e}")

    gold_path = run_pipeline(actors_paths, posts_paths, gold_dir)
    print("\n[main] ELT pipeline complete.\n")
    return gold_path


def run_stats(gold_path: str):
    if not os.path.exists(gold_path):
        print(f"[main] Gold not found at {gold_path}")
        return

    gold = pl.read_parquet(gold_path)
    er = gold["er_mean"].to_numpy()

    ineq = compute_inequality(er)
    anomaly = compute_anomalies(gold)
    breakdown = compute_breakdown(gold)

    print(summary_text(ineq, anomaly, breakdown))
    return ineq, anomaly, breakdown, gold


def main():
    parser = argparse.ArgumentParser(description="Attention Observatory ELT Pipeline")
    parser.add_argument("--skip-elt", action="store_true", help="Skip ingestion, use existing bronze")
    parser.add_argument("--simulate", action="store_true", help="Force simulated data")
    parser.add_argument("--gold-path", default="data/gold/fact_metrics.parquet")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard after pipeline")
    args = parser.parse_args()

    if args.skip_elt:
        bronze_dir = "data/bronze"
        actors_paths, posts_paths = _find_bronze_files(bronze_dir)
        print(f"[main] Found {len(actors_paths)} actor files, {len(posts_paths)} post files")

        if not actors_paths:
            print("[main] No existing bronze data found. Run without --skip-elt.")
            sys.exit(1)

        gold_dir = "data/gold"
        os.makedirs(gold_dir, exist_ok=True)
        gold_path = run_pipeline(actors_paths, posts_paths, gold_dir)
    else:
        gold_path = run_elt(force_simulator=args.simulate)

    result = run_stats(gold_path)
    if result:
        ineq, anomaly, breakdown, gold = result
        platforms = gold.group_by("platform").agg(pl.len().alias("n")).to_dict(as_series=False)
        plat_dict = dict(zip(platforms["platform"], platforms["n"]))
        from src.analysis.longitudinal import record_snapshot
        record_snapshot(
            gini=ineq.gini, alpha=ineq.powerlaw_alpha, sigma=ineq.powerlaw_sigma,
            n_super_hubs=anomaly.n_super_hubs, super_hub_share=anomaly.super_hub_attention_share,
            n_high_leverage=len(anomaly.high_leverage_nodes), churn=breakdown.churn_acceleration_mean,
            saturation=breakdown.systemic_saturation, n_actors=len(gold),
            n_posts=int(gold["post_count"].sum()), n_platforms=gold["platform"].n_unique(),
            platforms=plat_dict,
        )

    if args.dashboard:
        print("\n[main] Launching dashboard...")
        import subprocess
        subprocess.run(["streamlit", "run", "app.py"], check=False)
    else:
        print("\n[main] To launch dashboard:  streamlit run app.py")


if __name__ == "__main__":
    main()
