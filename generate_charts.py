import polars as pl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

DPI = 150
IMG = Path("img")
IMG.mkdir(exist_ok=True)

df = pl.read_parquet("data/gold/fact_metrics.parquet")
df_attention = df.filter(pl.col("is_attention_source"))
bronze_dir = Path("data/bronze")
post_files = list(bronze_dir.glob("*_posts.parquet"))
posts = pl.concat([pl.read_parquet(f) for f in post_files], how="diagonal") if post_files else pl.DataFrame()

platforms = df["platform"].unique().to_list()
colors = {
    "hackernews": "#FF6600",
    "wikipedia": "#000000",
    "huggingface": "#FBB03B",
    "bluesky": "#0085FF",
    "mastodon": "#6364FF",
    "github": "#2DA44E",
    "youtube": "#FF0000",
    "telegram": "#0088CC",
    "reddit": "#FF4500",
}
cmap = plt.colormaps["plasma"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "#0F1117",
    "axes.facecolor": "#161822",
    "axes.edgecolor": "#2A2D3A",
    "axes.labelcolor": "#E0E0E0",
    "text.color": "#E0E0E0",
    "xtick.color": "#888",
    "ytick.color": "#888",
    "legend.facecolor": "#161822",
    "legend.edgecolor": "#2A2D3A",
    "legend.labelcolor": "#E0E0E0",
    "grid.color": "#2A2D3A",
    "grid.alpha": 0.3,
})


def _lorenz(values):
    v = np.sort(values)
    n = len(v)
    if v.sum() == 0:
        return np.linspace(0, 1, n + 1), np.linspace(0, 1, n + 1)
    x = np.linspace(0, 1, n + 1)
    y = np.insert(np.cumsum(v, dtype=float) / v.sum(), 0, 0.0)
    return x, y


def _gini(values):
    v = np.sort(values)
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    return float((2 * np.sum(np.arange(1, n + 1) * v) - (n + 1) * v.sum()) / (n * v.sum()))


# ─── 1. Lorenz Curve ───
fig, ax = plt.subplots(figsize=(7, 7))
er = df_attention["er_mean"].to_numpy()
er = er[~np.isnan(er) & (er > 0)]
lx, ly = _lorenz(er)
ax.plot(lx, ly, color="#00D2FF", linewidth=2.5, label=f"Observed (Gini = {_gini(er):.3f})")
ax.plot([0, 1], [0, 1], "--", color="#555", linewidth=1.5, label="Perfect equality")
ax.fill_between(lx, ly, lx, alpha=0.15, color="#00D2FF")
g = _gini(er)
ax.annotate(f"Gini = {g:.3f}", xy=(0.7, 0.25), fontsize=16, color="#00D2FF",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#161822", edgecolor="#00D2FF"))
ax.set_xlabel("Cumulative share of actors")
ax.set_ylabel("Cumulative share of engagement")
ax.set_title("Engagement Concentration — Lorenz Curve")
ax.legend()
fig.tight_layout()
fig.savefig(IMG / "lorenz_curve.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ─── 2. Power Law (log-log) ───
fig, ax = plt.subplots(figsize=(7, 5))
er_sorted = np.sort(er)[::-1]
ccdf = np.arange(1, len(er_sorted) + 1) / len(er_sorted)
ax.loglog(er_sorted, ccdf, "o", markersize=2.5, color="#FF6B6B", alpha=0.7, label="Empirical CCDF")
er_pos = er_sorted[er_sorted > 0]
if len(er_pos) > 10:
    from scipy import stats as sp_stats
    log_er = np.log(er_pos)
    alpha_hat = 1 + len(er_pos) / np.sum(log_er - log_er.min())
    x_fit = np.logspace(np.log10(er_pos.min()), np.log10(er_pos.max()), 100)
    y_fit = (x_fit / er_pos.min()) ** (-alpha_hat + 1)
    y_fit = y_fit / y_fit.max() * ccdf.max()
    ax.loglog(x_fit, y_fit, "--", color="#FFD93D", linewidth=2, label=f"Power Law fit (α = {alpha_hat:.2f})")
ax.set_xlabel("Engagement Rate")
ax.set_ylabel("P(ER ≥ x)")
ax.set_title("Engagement Rate Distribution — Power Law")
ax.legend()
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()
fig.savefig(IMG / "powerlaw.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ─── 3. Gini by Platform ───
fig, ax = plt.subplots(figsize=(8, 5))
platform_gini = {}
for p in platforms:
    vals = df.filter(pl.col("platform") == p)["er_mean"].to_numpy()
    vals = vals[~np.isnan(vals) & (vals > 0)]
    if len(vals) > 0:
        platform_gini[p] = _gini(vals)
ps = sorted(platform_gini.keys(), key=lambda x: platform_gini[x], reverse=True)
vals = [platform_gini[p] for p in ps]
bars = ax.bar(range(len(ps)), vals, color=[colors.get(p, "#888") for p in ps], width=0.6, edgecolor="#2A2D3A", linewidth=1.2)
for i, (p, v) in enumerate(zip(ps, vals)):
    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#E0E0E0")
ax.set_xticks(range(len(ps)))
ax.set_xticklabels([p.title() for p in ps])
ax.set_ylabel("Gini Coefficient")
ax.set_title("Inequality by Platform")
ax.set_ylim(0, 1.1)
fig.tight_layout()
fig.savefig(IMG / "gini_by_platform.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ─── 4. Feature Space Scatter (ER vs PPI vs Sentiment) ───
fig = plt.figure(figsize=(10, 7))
gs = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0, 0], projection="3d")
pp = df.filter(pl.col("er_mean") < pl.col("er_mean").quantile(0.99)).to_pandas()
sc = ax.scatter(
    pp["ppi_mean"], pp["er_mean"], pp["sentiment_avg"],
    c=pp["afi_mean"], cmap="plasma", alpha=0.6, s=12, edgecolors="none"
)
cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label("AFI (Aspirational Framing)", color="#E0E0E0")
cbar.ax.yaxis.set_tick_params(color="#888")
plt.setp(plt.getp(cbar.ax, "yticklabels"), color="#888")
ax.set_xlabel("PPI (Production Pressure)")
ax.set_ylabel("ER (Engagement Rate)")
ax.set_zlabel("Sentiment")
ax.set_title("Feature Space: ER × PPI × Sentiment × AFI")
ax.xaxis.pane.set_facecolor("#1A1C2B")
ax.yaxis.pane.set_facecolor("#1A1C2B")
ax.zaxis.pane.set_facecolor("#1A1C2B")
fig.tight_layout()
fig.savefig(IMG / "feature_space_3d.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ─── 5. Top Actors ───
fig, ax = plt.subplots(figsize=(9, 6))
top = df.sort("er_mean", descending=True).head(20).to_pandas()
bars = ax.barh(range(len(top)), top["er_mean"].values, color=[colors.get(top["platform"].iloc[i], "#888") for i in range(len(top))], edgecolor="#2A2D3A", linewidth=0.8)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([f"{top['username'].iloc[i][:25]}" for i in range(len(top))], fontsize=8)
ax.set_xlabel("Engagement Rate (%)")
ax.set_title("Top 20 Actors by Engagement Rate")
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(IMG / "top_actors.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ─── 6. Cook's Distance (Leverage) ───
fig, ax = plt.subplots(figsize=(8, 5))
ppi = df["ppi_mean"].to_numpy()
er_v = df["er_mean"].to_numpy()
n = len(ppi)
X = np.column_stack([np.ones(n), ppi])
beta = np.linalg.lstsq(X, er_v, rcond=None)[0]
y_pred = X @ beta
residuals = er_v - y_pred
mse = np.sum(residuals ** 2) / max(1, n - 2)
try:
    leverage = np.diag(X @ np.linalg.inv(X.T @ X) @ X.T)
except np.linalg.LinAlgError:
    leverage = np.zeros(n)
cooks = (residuals ** 2 / (2 * mse + 1e-10)) * (leverage / ((1 - leverage) ** 2 + 1e-10))
cooks = np.nan_to_num(cooks, nan=0.0, posinf=0.0, neginf=0.0)
threshold = 4.0 / n
ax.scatter(ppi, er_v, s=10, c=cooks, cmap="inferno", alpha=0.7)
ax.axhline(er_v.mean(), color="#555", linestyle="--", linewidth=0.8, alpha=0.5)
ax.axvline(ppi.mean(), color="#555", linestyle="--", linewidth=0.8, alpha=0.5)
high_idx = cooks > threshold
ax.scatter(ppi[high_idx], er_v[high_idx], s=40, facecolors="none", edgecolors="#FFD93D", linewidths=1.5, label=f"High leverage ({high_idx.sum()})")
ax.text(0.02, 0.98, f"Cook's D threshold = {threshold:.4f}", transform=ax.transAxes, va="top", fontsize=9, color="#FF6B6B")
ax.set_xlabel("PPI (Production Pressure)")
ax.set_ylabel("ER (Engagement Rate)")
ax.set_title("Leverage Plot — Cook's Distance")
ax.legend()
fig.tight_layout()
fig.savefig(IMG / "cooks_distance.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ─── 7. Correlation Heatmap ───
from matplotlib.colors import LinearSegmentedColormap
fig, ax = plt.subplots(figsize=(6, 5.5))
num_df = df.select([
    "er_mean", "ppi_mean", "sentiment_avg", "afi_mean",
    "followers", "post_count", "total_interactions"
]).to_pandas()
corr = num_df.corr(method="spearman")
labels = ["ER", "PPI", "Sentiment", "AFI", "Followers", "Posts", "Interactions"]
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
for i in range(len(labels)):
    for j in range(len(labels)):
        val = corr.values[i, j]
        color = "white" if abs(val) > 0.5 else "#E0E0E0"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Spearman ρ", color="#E0E0E0")
cbar.ax.yaxis.set_tick_params(color="#888")
plt.setp(plt.getp(cbar.ax, "yticklabels"), color="#888")
ax.set_title("Feature Correlation Matrix")
fig.tight_layout()
fig.savefig(IMG / "correlation_heatmap.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# ─── 8. Banner ───
W, H = 1200, 300
fig = plt.figure(figsize=(W/DPI, H/DPI), dpi=DPI)
fig.patch.set_facecolor("#0A0B12")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

wave_x = np.linspace(0, W, 400)
ax.fill_between(wave_x, 0, H * 0.6 + H * 0.25 * np.sin(2 * np.pi * wave_x / 200), color="#161822", alpha=0.8)
ax.fill_between(wave_x, 0, H * 0.5 + H * 0.2 * np.sin(2 * np.pi * wave_x / 150 + 1), color="#1A1C2B", alpha=0.6)

peak_x = np.linspace(W * 0.15, W * 0.85, 300)
for i, (amp, freq, phase, color, alpha) in enumerate([
    (30, 180, 0, "#00D2FF", 0.3),
    (20, 120, 1, "#FF6B6B", 0.2),
    (15, 90, 2, "#FFD93D", 0.15),
]):
    ax.plot(peak_x, H * 0.5 + amp * np.sin(2 * np.pi * peak_x / freq + phase), color=color, linewidth=1.5, alpha=alpha)

ax.text(W * 0.08, H * 0.35, "ATTENTION", fontsize=48, fontweight="bold", color="#00D2FF", alpha=0.9)
ax.text(W * 0.08, H * 0.62, "OBSERVATORY", fontsize=32, fontweight="bold", color="#FF6B6B", alpha=0.85)
ax.text(W * 0.08, H * 0.80, "empirical modeling of digital attention ecology", fontsize=13, color="#888", alpha=0.8)
ax.text(W * 0.72, H * 0.88, "Gini 0.974 · α=2.11 · n=4,779 · 7 platforms", fontsize=9, color="#555", alpha=0.7)

nodes_x = [W * 0.75, W * 0.82, W * 0.88, W * 0.93, W * 0.79, W * 0.86, W * 0.91]
nodes_y = [H * 0.3, H * 0.25, H * 0.35, H * 0.2, H * 0.55, H * 0.5, H * 0.6]
for i, (nx, ny) in enumerate(zip(nodes_x, nodes_y)):
    radius = 5 if i < 4 else 3
    ax.scatter(nx, ny, s=radius**2, color="#00D2FF" if i < 2 else "#FF6B6B" if i < 4 else "#FFD93D", alpha=0.7, zorder=5)
    if i < 3:
        ax.plot([nx, nodes_x[i+1]], [ny, nodes_y[i+1]], color="#2A2D3A", linewidth=0.8, alpha=0.5)
ax.plot([W * 0.75, W * 0.79], [H * 0.3, H * 0.55], color="#2A2D3A", linewidth=0.8, alpha=0.5)
ax.plot([W * 0.82, W * 0.86], [H * 0.25, H * 0.5], color="#2A2D3A", linewidth=0.8, alpha=0.5)

fig.savefig(IMG / "banner.png", dpi=DPI, bbox_inches="tight", pad_inches=0)
plt.close(fig)

print(f"Generated {len(list(IMG.glob('*.png')))} charts in {IMG}/")
