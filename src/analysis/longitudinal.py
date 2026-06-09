"""Longitudinal tracking: records and loads execution snapshots."""

import json, os, glob
from datetime import datetime, timezone

EXECUTIONS_DIR = "data/executions"


def record_snapshot(gini: float, alpha: float, sigma: float, n_super_hubs: int,
                    super_hub_share: float, n_high_leverage: int, churn: float,
                    saturation: bool, n_actors: int, n_posts: int,
                    n_platforms: int, platforms: dict,
                    hhi: float | None = None, shannon: float | None = None,
                    effective_n: float | None = None, palma_ratio: float | None = None,
                    top1_share: float | None = None, rich_club: float | None = None) -> str:
    os.makedirs(EXECUTIONS_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc)
    snapshot = {
        "timestamp": ts.isoformat(),
        "ts_epoch": ts.timestamp(),
        "gini": round(gini, 4),
        "powerlaw_alpha": round(alpha, 4),
        "powerlaw_sigma": round(sigma, 4),
        "n_super_hubs": n_super_hubs,
        "super_hub_share": round(super_hub_share, 4),
        "n_high_leverage": n_high_leverage,
        "churn_acceleration": round(churn, 4),
        "systemic_saturation": saturation,
        "n_actors": n_actors,
        "n_posts": n_posts,
        "n_platforms": n_platforms,
        "platforms": platforms,
    }
    if hhi is not None:
        snapshot["hhi"] = round(hhi, 1)
    if shannon is not None:
        snapshot["shannon"] = round(shannon, 4)
    if effective_n is not None:
        snapshot["effective_n"] = round(effective_n, 1)
    if palma_ratio is not None:
        snapshot["palma_ratio"] = round(palma_ratio, 2)
    if top1_share is not None:
        snapshot["top1_share"] = round(top1_share, 6)
    if rich_club is not None:
        snapshot["rich_club"] = round(rich_club, 4)
    path = f"{EXECUTIONS_DIR}/{ts.strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)
    print(f"[longitudinal] Snapshot saved -> {path}")
    return path


def load_snapshots() -> list[dict]:
    os.makedirs(EXECUTIONS_DIR, exist_ok=True)
    files = sorted(glob.glob(f"{EXECUTIONS_DIR}/*.json"))
    snaps = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            snaps.append(json.load(fh))
    return snaps


def latest_snapshot() -> dict | None:
    snaps = load_snapshots()
    return snaps[-1] if snaps else None


def summary_table(snaps: list[dict]) -> list[dict]:
    rows = []
    for s in snaps:
        rows.append({
            "timestamp": s["timestamp"][:19].replace("T", " "),
            "gini": s["gini"],
            "alpha": s.get("powerlaw_alpha", 0),
            "super_hubs": s["n_super_hubs"],
            "hubs_share": f"{s['super_hub_share']:.1%}",
            "actors": s["n_actors"],
            "posts": s["n_posts"],
            "churn": s["churn_acceleration"],
            "saturation": "DETECTED" if s["systemic_saturation"] else "OK",
        })
    return rows
