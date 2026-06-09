"""Concentration metrics and inequality measures for attention distribution.

Extends the core inequality module with HHI, Shannon Entropy, Effective N,
Palma Ratio, Top Share, Rich Club, and Bootstrap confidence intervals.

Derived from the hypothesis framework:
  H1 — Power law in all domains
  H2 — Inequality scales with ecosystem size
  H3 — Super-hubs capture disproportionate share
  H6 — Structural (not psychological) property
"""

import numpy as np
import polars as pl
from typing import NamedTuple


class ConcentrationMetrics(NamedTuple):
    hhi: float
    shannon_entropy: float
    effective_n: float
    palma_ratio: float
    top1_share: float
    top5_share: float
    top10_share: float
    gini: float


def _shares(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    if total == 0:
        return np.zeros_like(values)
    return values / total


def compute_hhi(values: np.ndarray) -> float:
    p = _shares(values)
    return float(np.sum(p ** 2) * 10000)


def compute_shannon_entropy(values: np.ndarray) -> float:
    p = _shares(values)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def compute_effective_n(values: np.ndarray) -> float:
    p = _shares(values)
    return float(1.0 / np.sum(p ** 2))


def compute_palma_ratio(values: np.ndarray) -> float:
    sorted_v = np.sort(values)
    top10 = sorted_v[int(len(sorted_v) * 0.9):].sum()
    bottom40 = sorted_v[:int(len(sorted_v) * 0.4)].sum()
    if bottom40 == 0:
        return float("inf")
    return float(top10 / bottom40)


def compute_top_shares(values: np.ndarray) -> tuple[float, float, float]:
    sorted_v = np.sort(values)[::-1]
    total = sorted_v.sum()
    if total == 0:
        return 0.0, 0.0, 0.0
    n = len(sorted_v)
    top1 = int(max(1, n * 0.01))
    top5 = int(max(1, n * 0.05))
    top10 = int(max(1, n * 0.10))
    return (
        float(sorted_v[:top1].sum() / total),
        float(sorted_v[:top5].sum() / total),
        float(sorted_v[:top10].sum() / total),
    )


def compute_rich_club(values: np.ndarray, percentile: float = 90) -> float:
    threshold = np.percentile(values, percentile)
    rich = values[values >= threshold]
    if len(rich) < 2:
        return 0.0
    total = values.sum()
    if total == 0:
        return 0.0
    observed_share = rich.sum() / total
    uniform_share = len(rich) / len(values)
    if uniform_share == 0:
        return 0.0
    return float(observed_share / uniform_share)


def compute_concentration(values: np.ndarray) -> ConcentrationMetrics:
    values = values[~np.isnan(values) & (values > 0)]
    if len(values) < 2:
        return ConcentrationMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    gini = _gini(values)
    hhi = compute_hhi(values)
    entropy = compute_shannon_entropy(values)
    eff_n = compute_effective_n(values)
    palma = compute_palma_ratio(values)
    t1, t5, t10 = compute_top_shares(values)
    return ConcentrationMetrics(hhi, entropy, eff_n, palma, t1, t5, t10, gini)


def _gini(values: np.ndarray) -> float:
    v = np.sort(values)
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    return float((2 * np.sum(np.arange(1, n + 1) * v) - (n + 1) * v.sum()) / (n * v.sum()))


def bootstrap_gini(values: np.ndarray, n_iter: int = 1000, ci: float = 0.95) -> dict:
    values = values[~np.isnan(values) & (values > 0)]
    if len(values) < 10:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0}
    rng = np.random.default_rng(42)
    boot = np.array([
        _gini(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_iter)
    ])
    alpha = (1 - ci) / 2
    return {
        "mean": float(boot.mean()),
        "lower": float(np.percentile(boot, alpha * 100)),
        "upper": float(np.percentile(boot, (1 - alpha) * 100)),
        "std": float(boot.std()),
    }


def concentration_by_group(df: pl.DataFrame, group_col: str, value_col: str = "er_mean") -> pl.DataFrame:
    rows = []
    for group in df[group_col].unique():
        subset = df.filter(pl.col(group_col) == group)[value_col].to_numpy()
        subset = subset[~np.isnan(subset) & (subset > 0)]
        if len(subset) < 2:
            continue
        c = compute_concentration(subset)
        bg = bootstrap_gini(subset)
        rows.append({
            group_col: group,
            "n": len(subset),
            "gini": c.gini,
            "hhi": c.hhi,
            "shannon": c.shannon_entropy,
            "effective_n": c.effective_n,
            "palma_ratio": c.palma_ratio,
            "top1_share": c.top1_share,
            "top5_share": c.top5_share,
            "top10_share": c.top10_share,
            "gini_ci_lower": bg["lower"],
            "gini_ci_upper": bg["upper"],
        })
    return pl.DataFrame(rows)
