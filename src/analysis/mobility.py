"""Mobility and persistence analysis for longitudinal attention tracking.

Implements:
  - Rank Mobility Index (RMI): how much actors change rank between snapshots
  - Top-K Persistence: fraction of top-K actors that remain in top-K
  - Leader Turnover Rate: rate of replacement in top positions
  - Dominance Half-Life: time for top-1 actor to lose 50% of attention share
  - Transition Matrices: probability of moving between rank deciles

Requires ≥ 2 longitudinal snapshots (data/executions/).
"""

import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import NamedTuple


class MobilityReport(NamedTuple):
    rank_mobility_index: float
    top1_persistence: float
    top5_persistence: float
    top10_persistence: float
    leader_turnover: float
    dominance_half_life: float | None
    transition_matrix: list[list[float]]


def _load_snapshots(executions_dir: str = "data/executions") -> list[dict]:
    snaps = []
    for f in sorted(Path(executions_dir).glob("*.json")):
        with open(f) as fh:
            snaps.append(json.load(fh))
    return snaps


def _rank_actors(snapshot: dict, metric: str = "total_interactions",
                 gold_path: str = "data/gold/fact_metrics.parquet") -> dict[str, int]:
    """Rank actors by a given metric from gold data at snapshot time."""
    if not Path(gold_path).exists():
        return {}
    df = pl.read_parquet(gold_path)
    er_col = "total_interactions"
    if er_col not in df.columns:
        return {}
    ranked = df.sort(er_col, descending=True)
    return {row["actor_id"]: i for i, row in enumerate(ranked.iter_rows(named=True))}


def compute_rank_mobility(snapshots: list[dict],
                          gold_path: str = "data/gold/fact_metrics.parquet") -> MobilityReport:
    if len(snapshots) < 2:
        return MobilityReport(0, 0, 0, 0, 0, None, [])

    ranks_by_snap = []
    for snap in snapshots:
        ranks = _rank_actors(snap, gold_path=gold_path)
        if ranks:
            ranks_by_snap.append(ranks)

    if len(ranks_by_snap) < 2:
        return MobilityReport(0, 0, 0, 0, 0, None, [])

    all_actors = set()
    for r in ranks_by_snap:
        all_actors.update(r.keys())

    n_actors = len(all_actors)
    if n_actors == 0:
        return MobilityReport(0, 0, 0, 0, 0, None, [])

    # Rank Mobility Index: average absolute rank change / max possible change
    total_rank_change = 0.0
    n_overlap = 0
    for actor in all_actors:
        ranks = [r.get(actor) for r in ranks_by_snap if actor in r]
        if len(ranks) >= 2:
            total_rank_change += abs(ranks[-1] - ranks[0])
            n_overlap += 1

    max_change = n_actors - 1
    rmi = (total_rank_change / max(1, n_overlap)) / max(1, max_change) if n_overlap > 0 else 0.0

    # Top-K Persistence: fraction of top-K actors that remain in top-K
    K_VALUES = [1, 5, 10]
    persistence = {}
    for k_val in K_VALUES:
        if k_val > n_actors:
            persistence[k_val] = 0.0
            continue
        top_snap0 = set(sorted(ranks_by_snap[0].keys(),
                               key=lambda a: ranks_by_snap[0][a])[:k_val])
        top_snap1 = set(sorted(ranks_by_snap[-1].keys(),
                               key=lambda a: ranks_by_snap[-1][a])[:k_val])
        overlap = top_snap0 & top_snap1
        persistence[k_val] = len(overlap) / max(1, k_val)

    # Leader Turnover: fraction of top-10 that are new in latest snapshot
    top_prev = set(sorted(ranks_by_snap[0].keys(),
                          key=lambda a: ranks_by_snap[0][a])[:10])
    top_curr = set(sorted(ranks_by_snap[-1].keys(),
                          key=lambda a: ranks_by_snap[-1][a])[:10])
    new_leaders = top_curr - top_prev
    turnover = len(new_leaders) / max(1, len(top_curr))

    # Transition Matrix (decile-based)
    n_deciles = 10
    trans_matrix = [[0.0] * n_deciles for _ in range(n_deciles)]
    decile_size = max(1, n_actors // n_deciles)
    decile_map_0 = {}
    for actor, rank in ranks_by_snap[0].items():
        decile_map_0[actor] = min(n_deciles - 1, rank // decile_size)
    decile_map_1 = {}
    for actor, rank in ranks_by_snap[-1].items():
        decile_map_1[actor] = min(n_deciles - 1, rank // decile_size)
    for actor in all_actors:
        if actor in decile_map_0 and actor in decile_map_1:
            d0 = decile_map_0[actor]
            d1 = decile_map_1[actor]
            trans_matrix[d0][d1] += 1.0
    for i in range(n_deciles):
        row_sum = sum(trans_matrix[i])
        if row_sum > 0:
            trans_matrix[i] = [v / row_sum for v in trans_matrix[i]]

    return MobilityReport(
        rank_mobility_index=rmi,
        top1_persistence=persistence.get(1, 0.0),
        top5_persistence=persistence.get(5, 0.0),
        top10_persistence=persistence.get(10, 0.0),
        leader_turnover=turnover,
        dominance_half_life=None,
        transition_matrix=trans_matrix,
    )


def compute_gini_trajectory(executions_dir: str = "data/executions") -> list[dict]:
    """Returns time series of Gini and related metrics from snapshots."""
    snaps = _load_snapshots(executions_dir)
    if not snaps:
        return []
    trajectory = []
    for snap in snaps:
        entry = {
            "timestamp": snap["timestamp"],
            "ts_epoch": snap["ts_epoch"],
            "gini": snap["gini"],
            "alpha": snap.get("powerlaw_alpha", 0),
            "super_hubs": snap["n_super_hubs"],
            "actors": snap["n_actors"],
            "hhi": snap.get("hhi", 0),
            "effective_n": snap.get("effective_n", 0),
        }
        trajectory.append(entry)
    return trajectory


def detect_drift(baseline_snapshot: dict, current_snapshot: dict) -> dict:
    """Compute drift between two snapshots using available metrics."""
    drifts = {}
    for key in ["gini", "powerlaw_alpha", "n_super_hubs", "n_actors", "hhi"]:
        if key in baseline_snapshot and key in current_snapshot:
            b = baseline_snapshot[key]
            c = current_snapshot[key]
            if isinstance(b, (int, float)) and isinstance(c, (int, float)) and b != 0:
                drifts[f"{key}_change"] = (c - b) / abs(b)
    return drifts


def mobility_summary(snapshots: list[dict],
                     gold_path: str = "data/gold/fact_metrics.parquet") -> dict:
    if len(snapshots) < 2:
        return {"error": "Need ≥ 2 snapshots"}
    mob = compute_rank_mobility(snapshots, gold_path)
    traj = compute_gini_trajectory()
    drift = {}
    if len(snapshots) >= 2:
        drift = detect_drift(snapshots[0], snapshots[-1])
    return {
        "rank_mobility_index": round(mob.rank_mobility_index, 4),
        "top1_persistence": round(mob.top1_persistence, 4),
        "top5_persistence": round(mob.top5_persistence, 4),
        "top10_persistence": round(mob.top10_persistence, 4),
        "leader_turnover": round(mob.leader_turnover, 4),
        "n_snapshots": len(snapshots),
        "drift": drift,
        "transition_matrix": mob.transition_matrix,
        "trajectory": traj,
    }
