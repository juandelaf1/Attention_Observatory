import numpy as np
import polars as pl
from scipy import stats
from typing import NamedTuple


class InequalityMetrics(NamedTuple):
    gini: float
    lorenz_x: np.ndarray
    lorenz_y: np.ndarray
    powerlaw_alpha: float
    powerlaw_sigma: float
    powerlaw_is_pareto: bool


class ExpandedInequality(NamedTuple):
    gini: float
    hhi: float
    shannon_entropy: float
    effective_n: float
    palma_ratio: float
    top1_share: float
    top5_share: float
    top10_share: float
    rich_club: float
    gini_ci_lower: float
    gini_ci_upper: float


class AnomalyReport(NamedTuple):
    n_super_hubs: int
    super_hub_attention_share: float
    high_leverage_nodes: list[str]
    field_imbalance_count: int


class BreakdownMetrics(NamedTuple):
    churn_acceleration_mean: float
    systemic_saturation: bool
    isolation_ratio: float


def _safe_gini(values: np.ndarray) -> float:
    values = np.sort(values)
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    cumsum = np.cumsum(values, dtype=float)
    return float((2 * np.sum((np.arange(1, n + 1)) * values) - (n + 1) * values.sum()) / (n * values.sum()))


def _lorenz(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(values)
    n = len(values)
    if values.sum() == 0:
        return np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1)
    cumulative = np.cumsum(values, dtype=float)
    x = np.linspace(0, 1, n + 1)
    y = np.insert(cumulative / cumulative[-1], 0, 0.0)
    return x, y


def _powerlaw_fit(values: np.ndarray) -> tuple[float, float, bool]:
    values = values[values > 0]
    if len(values) < 10:
        return 0.0, 0.0, False
    try:
        import powerlaw
        fit = powerlaw.Fit(values, verbose=False)
        alpha = fit.power_law.alpha
        sigma = fit.power_law.sigma
        R, p = fit.distribution_compare("power_law", "lognormal")
        return float(alpha), float(sigma), R > 0
    except Exception:
        alpha = 1 + len(values) / np.sum(np.log(values / values.min()))
        sigma = alpha / np.sqrt(len(values))
        return float(alpha), float(sigma), alpha > 2.0


def _cooks_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(x)
    if n < 3:
        return np.zeros(n)
    try:
        X = np.column_stack([np.ones(n), x])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta
        residuals = y - y_pred
        mse = np.sum(residuals ** 2) / max(1, n - 2)
        leverage = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
        d = (residuals ** 2 / (2 * mse + 1e-10)) * (leverage / ((1 - leverage) ** 2 + 1e-10))
        return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    except np.linalg.LinAlgError:
        return np.zeros(n)


def _shares(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    if total == 0:
        return np.zeros_like(values)
    return values / total


def bootstrap_gini_ci(values: np.ndarray, n_iter: int = 1000) -> tuple[float, float]:
    values = values[~np.isnan(values) & (values > 0)]
    if len(values) < 10:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    boot = np.array([
        _safe_gini(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_iter)
    ])
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def compute_expanded_inequality(values: np.ndarray) -> ExpandedInequality:
    values = values[~np.isnan(values) & (values > 0)]
    if len(values) < 2:
        return ExpandedInequality(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    gini = _safe_gini(values)
    p = _shares(values)
    hhi = float(np.sum(p ** 2) * 10000)
    p_pos = p[p > 0]
    entropy = float(-np.sum(p_pos * np.log(p_pos))) if len(p_pos) > 0 else 0.0
    eff_n = float(1.0 / np.sum(p ** 2))
    sorted_v = np.sort(values)
    top10_val = sorted_v[int(len(sorted_v) * 0.9):].sum()
    bottom40_val = sorted_v[:int(len(sorted_v) * 0.4)].sum()
    palma = float(top10_val / bottom40_val) if bottom40_val > 0 else float("inf")
    sorted_desc = np.sort(values)[::-1]
    total = sorted_desc.sum()
    n = len(sorted_desc)
    t1 = sorted_desc[:max(1, n // 100)].sum() / total if total > 0 else 0
    t5 = sorted_desc[:max(1, n // 20)].sum() / total if total > 0 else 0
    t10 = sorted_desc[:max(1, n // 10)].sum() / total if total > 0 else 0
    threshold = np.percentile(values, 90)
    rich = values[values >= threshold]
    rich_club = float(rich.sum() / total / (len(rich) / max(1, len(values)))) if len(rich) >= 2 and total > 0 else 0.0
    ci_low, ci_high = bootstrap_gini_ci(values)
    return ExpandedInequality(gini, hhi, entropy, eff_n, palma, t1, t5, t10, rich_club, ci_low, ci_high)


def compute_inequality(values: np.ndarray) -> InequalityMetrics:
    gini = _safe_gini(values)
    lx, ly = _lorenz(values)
    alpha, sigma, pareto = _powerlaw_fit(values)
    return InequalityMetrics(gini, lx, ly, alpha, sigma, pareto)


def compute_anomalies(df: pl.DataFrame) -> AnomalyReport:
    er = df["er_mean"].to_numpy()
    ppi = df["ppi_mean"].to_numpy()
    sent_var = df["sentiment_variance"].to_numpy()

    er_z = np.abs(stats.zscore(er, nan_policy="omit"))
    super_hub_mask = er_z > 3
    n_super = int(super_hub_mask.sum())

    total_interactions = df["total_interactions"].to_numpy().sum()
    super_hub_interactions = df.filter(pl.col("actor_id").is_in(
        df.filter(pl.col("er_mean") > (pl.col("er_mean").mean() + 3 * pl.col("er_mean").std()) )["actor_id"]
    ))["total_interactions"].sum() if total_interactions > 0 else 0
    super_share = float(super_hub_interactions) / float(total_interactions) if total_interactions > 0 else 0.0

    cooks = _cooks_distance(ppi, er)
    high_leverage_mask = cooks > 4.0 / max(1, len(cooks))
    high_leverage_ids = df.filter(pl.Series("_cooks", cooks) > 4.0 / max(1, len(cooks)) )["actor_id"].to_list()

    imbalance_mask = (ppi > ppi.mean() + ppi.std()) & (er < er.mean() - er.std())
    n_imbalance = int(imbalance_mask.sum())

    return AnomalyReport(n_super, super_share, high_leverage_ids, n_imbalance)


def compute_breakdown(actors: pl.DataFrame, posts: pl.DataFrame | None = None) -> BreakdownMetrics:
    er = actors["er_mean"].to_numpy()
    ppi = actors["ppi_mean"].to_numpy()

    churn_accel = np.gradient(np.nan_to_num(er)) * np.gradient(np.nan_to_num(ppi))
    churn_mean = float(np.mean(np.abs(churn_accel)))

    gini_val = _safe_gini(er)
    global_er_drop = float(np.mean(er)) < 0.1 * float(np.mean(ppi)) if float(np.mean(ppi)) > 0 else False
    systemic_sat = bool(gini_val > 0.7 and global_er_drop)

    isolation = 0.0
    if posts is not None:
        total_interactions = (posts["likes"].sum() + posts["comments"].sum() + posts["shares"].sum())
        if total_interactions > 0:
            n_actors = posts["actor_id"].n_unique()
            if n_actors > 1:
                actor_totals = posts.group_by("actor_id").agg(
                    (pl.col("likes") + pl.col("comments") + pl.col("shares")).sum().alias("actor_total")
                )
                top_share = actor_totals.sort("actor_total", descending=True).head(max(1, n_actors // 10))["actor_total"].sum()
                isolation = float(top_share) / float(total_interactions)

    return BreakdownMetrics(churn_mean, systemic_sat, isolation)


def summary_text(ineq: InequalityMetrics, anomaly: AnomalyReport, breakdown: BreakdownMetrics) -> str:
    lines = [
        "=" * 60,
        "ATTENTION OBSERVATORY — STATISTICAL SUMMARY",
        "=" * 60,
        "",
        f"Gini Coefficient:               {ineq.gini:.4f}",
        f"Power Law Alpha:                {ineq.powerlaw_alpha:.4f}",
        f"Power Law Sigma:                {ineq.powerlaw_sigma:.4f}",
        f"Pareto Distribution:            {'YES' if ineq.powerlaw_is_pareto else 'NO'}",
        "",
        f"Super-Hubs (Z > 3):             {anomaly.n_super_hubs}",
        f"Attention share of Super-Hubs:  {anomaly.super_hub_attention_share:.2%}",
        f"High-Leverage Nodes:            {len(anomaly.high_leverage_nodes)}",
        f"Field Imbalance Zones:          {anomaly.field_imbalance_count}",
        "",
        f"Churn Acceleration (mean):      {breakdown.churn_acceleration_mean:.4f}",
        f"Systemic Saturation:            {'DETECTED' if breakdown.systemic_saturation else 'NOT DETECTED'}",
        f"Subgraph Isolation Ratio:       {breakdown.isolation_ratio:.4f}",
        "",
    ]
    if breakdown.systemic_saturation:
        lines += [
            "-" * 60,
            "SYSTEMIC BREAKDOWN WARNING",
            "The ecosystem has reached its evolutionary limit.",
            "Attention extraction rate destroys the cognitive capacity of the resource.",
            "Engagement decline is not content scarcity — it is biological homeostasis.",
            "-" * 60,
        ]
    if anomaly.n_super_hubs > 0:
        lines += [
            "",
            "NOTE: High-leverage points confirm network vulnerability.",
            "When super-hub attention load exceeds agent stability threshold,",
            "node behavior enters erratic regimes, altering residual attention",
            "availability for the aspirant base.",
        ]
    return "\n".join(lines)
