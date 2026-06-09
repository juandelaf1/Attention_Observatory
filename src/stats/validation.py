"""Statistical validation tests for the attention observatory.

Implements:
  - Power Law vs Lognormal (Likelihood Ratio Test via powerlaw library)
  - Bootstrap confidence intervals for concentration metrics
  - Kruskal-Wallis H-test for multi-group comparison
  - Dunn post-hoc test with Bonferroni correction
  - Partial Spearman correlation (controlling for audience size)

Validates hypotheses H1–H6 from the research framework.
"""

import numpy as np
import polars as pl
from scipy import stats
from typing import NamedTuple


class PowerLawTest(NamedTuple):
    alpha: float
    sigma: float
    xmin: float
    loglikelihood_ratio: float
    p_value: float
    is_pareto: bool
    best_model: str  # "power_law", "lognormal", "exponential", "unknown"


def test_powerlaw(values: np.ndarray, compare_against: list[str] | None = None) -> PowerLawTest:
    values = values[~np.isnan(values) & (values > 0)]
    if len(values) < 10:
        return PowerLawTest(0, 0, 0, 0, 1, False, "unknown")
    try:
        import powerlaw
        fit = powerlaw.Fit(values, verbose=False, xmin=None)
        alpha = fit.power_law.alpha
        sigma = fit.power_law.sigma
        xmin = fit.power_law.xmin
        if compare_against is None:
            compare_against = ["lognormal", "exponential"]
        best = "power_law"
        best_llr = 0
        best_p = 1.0
        for alt in compare_against:
            try:
                R, p = fit.distribution_compare("power_law", alt)
                if p < 0.05 and R > best_llr:
                    best_llr = R
                    best_p = p
                    best = "power_law"
                elif p < 0.05 and R < 0:
                    best = alt
                    best_llr = R
                    best_p = p
            except Exception:
                continue
        return PowerLawTest(
            alpha=float(alpha), sigma=float(sigma), xmin=float(xmin),
            loglikelihood_ratio=float(best_llr), p_value=float(best_p),
            is_pareto=alpha > 2.0, best_model=best
        )
    except Exception:
        alpha_l = 1 + len(values) / np.sum(np.log(values / values.min()))
        sigma_l = alpha_l / np.sqrt(len(values))
        return PowerLawTest(
            alpha=float(alpha_l), sigma=float(sigma_l), xmin=float(values.min()),
            loglikelihood_ratio=0, p_value=1, is_pareto=alpha_l > 2.0,
            best_model="unknown"
        )


def bootstrap_ci(values: np.ndarray, metric_fn, n_iter: int = 1000, ci: float = 0.95) -> dict:
    values = values[~np.isnan(values) & (values > 0)]
    if len(values) < 10:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0}
    rng = np.random.default_rng(42)
    boot = np.array([
        metric_fn(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_iter)
    ])
    alpha = (1 - ci) / 2
    return {
        "mean": float(boot.mean()),
        "lower": float(np.percentile(boot, alpha * 100)),
        "upper": float(np.percentile(boot, (1 - alpha) * 100)),
        "std": float(boot.std()),
    }


def kruskal_wallis_report(df: pl.DataFrame, group_col: str, value_col: str = "er_mean") -> dict:
    groups = {}
    for group in df[group_col].unique():
        vals = df.filter(pl.col(group_col) == group)[value_col].to_numpy()
        vals = vals[~np.isnan(vals) & (vals > 0)]
        if len(vals) > 2:
            groups[str(group)] = vals
    if len(groups) < 2:
        return {"h_stat": 0.0, "p_value": 1.0, "n_groups": len(groups), "groups": list(groups.keys())}
    group_names = list(groups.keys())
    group_vals = [groups[g] for g in group_names]
    h, p = stats.kruskal(*group_vals)
    return {"h_stat": float(h), "p_value": float(p), "n_groups": len(groups), "groups": group_names}


def dunn_posthoc(df: pl.DataFrame, group_col: str, value_col: str = "er_mean") -> list[dict]:
    try:
        import scikit_posthocs as sp
    except ImportError:
        return [{"error": "scikit_posthocs not installed. pip install scikit-posthocs"}]
    pdf = df.select([group_col, value_col]).to_pandas().dropna()
    if pdf[value_col].dtype != "float64":
        pdf[value_col] = pdf[value_col].astype(float)
    try:
        result = sp.posthoc_dunn(pdf, val_col=value_col, group_col=group_col, p_adjust="bonferroni")
        pairs = []
        for i, row_name in enumerate(result.index):
            for j, col_name in enumerate(result.columns):
                if i < j:
                    pairs.append({
                        "group_a": row_name,
                        "group_b": col_name,
                        "p_value": float(result.iloc[i, j]),
                        "significant": result.iloc[i, j] < 0.05
                    })
        return pairs
    except Exception as e:
        return [{"error": str(e)}]


def partial_spearman(df: pl.DataFrame, x_col: str, y_col: str, control_col: str) -> dict:
    try:
        from pingouin import partial_corr
    except ImportError:
        return {"error": "pingouin not installed. pip install pingouin"}
    pdf = df.select([x_col, y_col, control_col]).to_pandas().dropna()
    try:
        result = partial_corr(data=pdf, x=x_col, y=y_col, covar=control_col, method="spearman")
        return {
            "r": float(result["r"].iloc[0]),
            "p_val": float(result["p-val"].iloc[0]),
            "n": len(pdf),
        }
    except Exception as e:
        return {"error": str(e)}


def longitudinal_ks_test(baseline: np.ndarray, current: np.ndarray) -> dict:
    stat, p = stats.ks_2samp(baseline, current)
    return {"ks_statistic": float(stat), "p_value": float(p), "drift_detected": p < 0.05}


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    expected_pct, _ = np.histogram(expected, bins=bins, range=(0, np.percentile(expected, 99)), density=True)
    actual_pct, _ = np.histogram(actual, bins=bins, range=(0, np.percentile(expected, 99)), density=True)
    expected_pct = expected_pct / expected_pct.sum()
    actual_pct = actual_pct / actual_pct.sum()
    expected_pct = np.clip(expected_pct, 1e-10, None)
    actual_pct = np.clip(actual_pct, 1e-10, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
