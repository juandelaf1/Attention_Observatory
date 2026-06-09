import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import polars as pl
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.stats.inequality import compute_inequality, compute_expanded_inequality, compute_anomalies, compute_breakdown, _lorenz, _cooks_distance
from src.stats.network import compute_network_metrics, compute_cross_platform_report
from src.stats.concentration import concentration_by_group

st.set_page_config(page_title="Attention Observatory", page_icon="", layout="wide")

C1, C2, C3, C4, C5 = "#E74C3C", "#3498DB", "#E67E22", "#2ECC71", "#9B59B6"
BG = "#FAFAFA"

@st.cache_data
def load_gold(p="data/gold/fact_metrics.parquet"):
    return pl.read_parquet(p)

@st.cache_data
def load_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def compute_metrics(df):
    er = df["er_mean"].to_numpy()
    return compute_inequality(er), compute_anomalies(df), compute_breakdown(df)

# ── Charts ────────────────────────────────────────────────

def lorenz_curve(ineq):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect Equality", line=dict(dash="dash", color=C2, width=1.5)))
    fig.add_trace(go.Scatter(x=ineq.lorenz_x, y=ineq.lorenz_y, mode="lines", name=f"Lorenz (Gini={ineq.gini:.4f})", fill="tozeroy", line=dict(color=C1, width=3)))
    fig.add_annotation(x=0.25, y=0.85, text=f"<b>Gini = {ineq.gini:.4f}</b>", showarrow=False, font=dict(size=16, color=C1), bgcolor="rgba(255,255,255,0.85)", bordercolor=C1, borderwidth=1, borderpad=6)
    fig.update_layout(title="Lorenz Curve — Attention Concentration", xaxis=dict(title="Cumulative Actors", tickformat=".0%", range=[0,1]), yaxis=dict(title="Cumulative Attention", tickformat=".0%", range=[0,1]), plot_bgcolor=BG, margin=dict(l=40,r=20,t=50,b=40))
    return fig

def er_hist(er_arr):
    fig = px.histogram(x=er_arr, nbins=40, log_y=True, title="Engagement Distribution (log)", labels={"x": "ER (%)"}, color_discrete_sequence=[C1])
    fig.update_traces(opacity=0.75)
    fig.update_layout(plot_bgcolor=BG, margin=dict(l=40,r=20,t=50,b=40), showlegend=False)
    return fig

def platform_bar(filtered):
    counts = filtered.group_by("platform").len().sort("len", descending=True).to_pandas()
    fig = px.bar(counts, x="platform", y="len", title="Actors by Platform", color="platform", color_discrete_sequence=px.colors.qualitative.Set2, text="len", labels={"len":"Actors","platform":""})
    fig.update_traces(textposition="outside")
    fig.update_layout(plot_bgcolor=BG, showlegend=False, margin=dict(l=40,r=20,t=50,b=40))
    return fig

def top_actors(filtered, n=20):
    top = filtered.sort("er_mean", descending=True).head(n).to_pandas()
    top = top.sort_values("er_mean")
    fig = px.bar(top, y="actor_id", x="er_mean", orientation="h", title=f"Top {n} by Engagement", labels={"er_mean":"ER (%)", "actor_id":""}, color="platform", hover_data={"max_followers":":,","ppi_mean":":.3f","afi_mean":":.4f"})
    fig.update_layout(plot_bgcolor=BG, margin=dict(l=10,r=20,t=50,b=40), height=500, yaxis=dict(autorange="reversed"), showlegend=True, legend=dict(orientation="h", y=1.02))
    return fig

def cooks_plot(df_pd):
    er, ppi = df_pd["er_mean"].values, df_pd["ppi_mean"].values
    cooks = _cooks_distance(ppi, er)
    thr = 4.0 / max(1, len(cooks))
    df = df_pd.copy()
    df["cooks_d"] = cooks
    df["is_high"] = cooks > thr
    fig = go.Figure()
    for is_h, name, col in [(False,"Normal",C2),(True,"High-Leverage",C1)]:
        sub = df[df["is_high"] == is_h]
        if sub.empty: continue
        fig.add_trace(go.Scatter(x=sub["ppi_mean"], y=sub["er_mean"], mode="markers", name=name, marker=dict(color=col, size=6+15*sub["cooks_d"]/max(1,sub["cooks_d"].max()), symbol="circle" if not is_h else "star", opacity=0.7 if not is_h else 0.9, line=dict(width=0.5,color="rgba(0,0,0,0.2)")), hovertemplate="<b>%{customdata[0]}</b><br>PPI: %{x:.3f}<br>ER: %{y:.2f}%<br>Cook's D: %{customdata[1]:.4f}<extra></extra>", customdata=np.column_stack([sub["actor_id"].values, sub["cooks_d"].values])))
    fig.add_hline(y=thr, line_dash="dash", line_color=C1, annotation_text=f"threshold={thr:.4f}", annotation_position="bottom right")
    fig.update_layout(title=f"Leverage Plot — {df['is_high'].sum()} High-Leverage Nodes", xaxis_title="PPI", yaxis_title="ER (%)", plot_bgcolor=BG, margin=dict(l=40,r=20,t=50,b=40))
    return fig

def gini_by_platform_bar(p5):
    df = pd.DataFrame([{ "platform": k, "gini": v["gini"], "n": v["n"] } for k,v in p5.items() if v["gini"] is not None and v["gini"] > 0])
    if df.empty: return None
    df = df.sort_values("gini")
    fig = px.bar(df, y="platform", x="gini", orientation="h", title="Gini by Platform", text="gini", color="gini", color_continuous_scale="Reds", labels={"gini":"Gini","platform":""})
    fig.update_traces(texttemplate="%{x:.4f}", textposition="outside")
    fig.update_layout(plot_bgcolor=BG, margin=dict(l=40,r=20,t=50,b=40), showlegend=False)
    return fig

def powerlaw_plot(filtered, ineq):
    valid = filtered.filter((pl.col("er_mean") > 0) & (pl.col("max_followers") > 0)).to_pandas()
    if len(valid) < 10: return None
    x, y = np.log10(valid["max_followers"].values), np.log10(valid["er_mean"].values)
    slope, intercept = np.polyfit(x, y, 1)
    x_dom = np.logspace(np.log10(valid["max_followers"].min()), np.log10(valid["max_followers"].max()), 100)
    y_line = 10 ** (intercept + slope * np.log10(x_dom))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=valid["max_followers"], y=valid["er_mean"], mode="markers", name=f"Actors ({len(valid)})", marker=dict(color=C2, size=6, opacity=0.55), hovertemplate="<b>%{customdata}</b><br>Followers: %{x:,.0f}<br>ER: %{y:.2f}%<extra></extra>", customdata=valid["actor_id"].values))
    fig.add_trace(go.Scatter(x=x_dom, y=y_line, mode="lines", name=f"Power Law (alpha={ineq.powerlaw_alpha:.3f})", line=dict(color=C5, width=2, dash="dash")))
    fig.update_layout(title=f"Power Law — Engagement x Followers (alpha={ineq.powerlaw_alpha:.3f})", xaxis=dict(title="Followers", type="log"), yaxis=dict(title="ER (%)", type="log"), plot_bgcolor=BG, margin=dict(l=40,r=20,t=50,b=40))
    return fig

# ── Main ──────────────────────────────────────────────────

def main():
    df = load_gold()
    ineq, anomaly, breakdown = compute_metrics(df)
    p1 = load_json("data/reports/research/p1_powerlaw.json")
    p2 = load_json("data/reports/research/p2_sentiment_er.json")
    p3 = load_json("data/reports/research/p3_leverage.json")
    p4 = load_json("data/reports/research/p4_afi_ppi.json")
    p5 = load_json("data/reports/research/p5_gini_platform.json")
    p6 = load_json("data/reports/research/p6_superhubs.json")

    st.title("Attention Observatory")
    st.markdown("---")

    platforms = df["platform"].unique().to_list()
    selected_platforms = st.sidebar.multiselect("Platform", platforms, default=platforms)
    only_attention = st.sidebar.checkbox("Only attention sources (excl. HuggingFace)", value=True)

    domains = df["domain"].unique().to_list()
    selected_domains = st.sidebar.multiselect("Domain", domains, default=domains)

    f_min, f_max = int(df["max_followers"].min()), int(df["max_followers"].max())
    fol_range = st.sidebar.slider("Min followers", f_min, f_max, (f_min, f_max), step=1)

    mask = (pl.col("max_followers") >= fol_range[0]) & (pl.col("max_followers") <= fol_range[1])
    if selected_platforms:
        mask &= pl.col("platform").is_in(selected_platforms)
    if selected_domains:
        mask &= pl.col("domain").is_in(selected_domains)
    if only_attention:
        mask &= pl.col("is_attention_source")

    filtered = df.filter(mask)
    fpd = filtered.to_pandas()

    from src.analysis.longitudinal import load_snapshots, summary_table
    snaps = load_snapshots()

    tabs = st.tabs(["Overview", "Inequality", "Research", "Longitudinal", "State Space", "Actors", "Network"])

    # ── TAB 1: Overview ──
    with tabs[0]:
        st.header("System Overview")
        er_vals = filtered["er_mean"].to_numpy()
        con = compute_expanded_inequality(er_vals)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Actors", len(filtered))
        c2.metric("Posts", int(filtered["post_count"].sum()))
        c3.metric("Platforms", len(filtered["platform"].unique()))
        c4.metric("Domains", len(filtered["domain"].unique()))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gini", f"{con.gini:.4f}")
        c2.metric("HHI", f"{con.hhi:.1f}")
        c3.metric("Shannon Entropy", f"{con.shannon_entropy:.2f}")
        c4.metric("Effective N", f"{con.effective_n:.1f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Top 1% Share", f"{con.top1_share:.1%}")
        c2.metric("Top 10% Share", f"{con.top10_share:.1%}")
        c3.metric("Palma Ratio", f"{con.palma_ratio:.1f}")
        c4.metric("Rich Club (p90)", f"{con.rich_club:.2f}")

        r1 = st.columns(2)
        with r1[0]: st.plotly_chart(platform_bar(filtered), use_container_width=True)
        with r1[1]:
            domain_counts = filtered.group_by("domain").len().sort("len", descending=True).to_pandas()
            fig = px.bar(domain_counts, x="domain", y="len", title="Actors by Domain", color="domain", text="len", labels={"len":"Actors","domain":""})
            fig.update_traces(textposition="outside")
            fig.update_layout(plot_bgcolor=BG, showlegend=False, margin=dict(l=40,r=20,t=50,b=40))
            st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("External Ecosystem", f"{filtered['has_external_ecosystem'].sum() / max(1,len(filtered)) * 100:.1f}%")
        c2.metric("Legally Truncated", int(filtered["is_legally_truncated"].sum()))
        c3.metric("Prestige Drift", int(filtered["prestige_drift_detected"].sum()))

        if breakdown.systemic_saturation:
            st.error(f"Systemic Saturation — Gini={ineq.gini:.4f}  Churn={breakdown.churn_acceleration_mean:.2f}")

        st.plotly_chart(top_actors(filtered), use_container_width=True)

    # ── TAB 2: Inequality ──
    with tabs[1]:
        st.header("Attention Inequality")
        er_vals = filtered["er_mean"].to_numpy()
        con = compute_expanded_inequality(er_vals)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gini", f"{con.gini:.4f}")
        c2.metric("Gini CI 95%", f"({con.gini_ci_lower:.4f}, {con.gini_ci_upper:.4f})")
        c3.metric("Power Law alpha", f"{ineq.powerlaw_alpha:.4f}")
        c4.metric("Super-Hubs", f"{anomaly.n_super_hubs} ({anomaly.super_hub_attention_share:.1%})")

        r1 = st.columns([3, 2])
        with r1[0]: st.plotly_chart(lorenz_curve(ineq), use_container_width=True)
        with r1[1]: st.plotly_chart(er_hist(er_vals), use_container_width=True)

        st.subheader("Concentration Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("HHI", f"{con.hhi:.1f}")
        c2.metric("Shannon Entropy", f"{con.shannon_entropy:.2f}")
        c3.metric("Effective N", f"{con.effective_n:.1f}")
        c4.metric("Palma Ratio", f"{con.palma_ratio:.1f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Top 1% Share", f"{con.top1_share:.1%}")
        c2.metric("Top 5% Share", f"{con.top5_share:.1%}")
        c3.metric("Top 10% Share", f"{con.top10_share:.1%}")
        c4.metric("Rich Club (p90)", f"{con.rich_club:.2f}")

        pl_plot = powerlaw_plot(filtered, ineq)
        if pl_plot: st.plotly_chart(pl_plot, use_container_width=True)

        gbar = gini_by_platform_bar(p5)
        if gbar: st.plotly_chart(gbar, use_container_width=True)

        st.subheader("Concentration by Domain")
        domain_con = concentration_by_group(filtered, "domain", "er_mean")
        if len(domain_con) > 0:
            st.dataframe(domain_con.to_pandas(), use_container_width=True, hide_index=True)

    # ── TAB 3: Research ──
    with tabs[2]:
        st.header("Research Findings")

        r1, r2 = st.columns(2)
        with r1:
            st.subheader("P1: Power Law by Platform")
            p1d = pd.DataFrame([{ "platform": k, **{kk:vv for kk,vv in v.items() if kk in ("alpha","n_with_er","n")} } for k,v in p1.items() if v.get("alpha") is not None])
            if not p1d.empty:
                p1d = p1d.sort_values("alpha", ascending=False)
                fig = px.bar(p1d, x="platform", y="alpha", title="Power Law alpha by Platform", text="alpha", color="alpha", color_continuous_scale="Viridis", labels={"alpha":"alpha","platform":""})
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig.update_layout(showlegend=False, plot_bgcolor=BG, margin=dict(l=40,r=20,t=50,b=40))
                st.plotly_chart(fig, use_container_width=True)
            st.caption("alpha > 2 confirms heavy-tail (Pareto). HackerNews and Bluesky follow power law.")

        with r2:
            st.subheader("P2: Sentiment vs Engagement")
            c1, c2 = st.columns(2)
            c1.metric("Spearman rho", f"{p2['spearman_rho']:.4f}")
            c2.metric("p-value", f"{p2['spearman_p']:.4f}")
            st.caption("No correlation. Content tone does not predict engagement.")

        r3, r4 = st.columns(2)
        with r3:
            st.subheader("P4: AFI vs PPI")
            c1, c2 = st.columns(2)
            c1.metric("Spearman rho", f"{p4['spearman_rho']:.4f}")
            c2.metric("p-value", f"{p4['spearman_p']:.4f}")
            st.caption("Weak negative correlation. Prestige language increases as production pressure decreases.")
        with r4:
            st.subheader("P5: Gini by Platform")
            p5d = pd.DataFrame([{ "platform": k, "gini": v["gini"], "n": v["n"] } for k,v in p5.items() if v["gini"] is not None and v["gini"] > 0])
            if not p5d.empty:
                p5d = p5d.sort_values("gini")
                fig = px.bar(p5d, y="platform", x="gini", orientation="h", title="Gini by Platform", text="gini", color="gini", color_continuous_scale="Reds", labels={"gini":"Gini","platform":""})
                fig.update_traces(texttemplate="%{x:.4f}", textposition="outside")
                fig.update_layout(showlegend=False, plot_bgcolor=BG, margin=dict(l=40,r=20,t=50,b=40))
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("P6: Super-Hub Profile")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Super-Hubs", p6["n_hubs"])
        c2.metric("Mean ER", f"{p6['mean_vector']['er']:.1f}")
        c3.metric("Mean PPI", f"{p6['mean_vector']['ppi']:.3f}")
        c4.metric("Mean AFI", f"{p6['mean_vector']['afi']:.4f}")
        if p6["top_hubs"]:
            st.dataframe(pd.DataFrame(p6["top_hubs"]).rename(columns={"username":"User","platform":"Platform","er":"ER","ppi":"PPI"}),
                        column_config={"ER": st.column_config.NumberColumn(format="%.1f"), "PPI": st.column_config.NumberColumn(format="%.2f")},
                        use_container_width=True, hide_index=True)

    # ── TAB 4: Longitudinal ──
    with tabs[3]:
        st.header("Longitudinal Analysis")
        if len(snaps) < 2:
            st.info("Need at least 2 execution snapshots for longitudinal tracking. Run `python main.py` multiple times.")
            if snaps:
                st.json(snaps[-1])
        else:
            df_snaps = pd.DataFrame(summary_table(snaps))
            df_snaps["exec"] = range(1, len(df_snaps) + 1)
            st.dataframe(df_snaps, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.line(df_snaps, x="exec", y="gini", markers=True, title="Gini Over Time",
                             labels={"exec": "Execution", "gini": "Gini"}, text="gini")
                fig.update_traces(texttemplate="%{text:.4f}", textposition="top center")
                fig.update_layout(plot_bgcolor=BG, margin=dict(l=40,r=20,t=50,b=40))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.line(df_snaps, x="exec", y="alpha", markers=True, title="Power Law Alpha Over Time",
                             labels={"exec": "Execution", "alpha": "Alpha"}, text="alpha")
                fig.update_traces(texttemplate="%{text:.3f}", textposition="top center")
                fig.update_layout(plot_bgcolor=BG, margin=dict(l=40,r=20,t=50,b=40))
                st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.line(df_snaps, x="exec", y="super_hubs", markers=True, title="Super-Hubs Over Time",
                             labels={"exec": "Execution", "super_hubs": "Super-Hubs"}, text="super_hubs")
                fig.update_traces(textposition="top center")
                fig.update_layout(plot_bgcolor=BG, margin=dict(l=40,r=20,t=50,b=40))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.line(df_snaps, x="exec", y="actors", markers=True, title="Actors Over Time",
                             labels={"exec": "Execution", "actors": "Actors"}, text="actors")
                fig.update_traces(textposition="top center")
                fig.update_layout(plot_bgcolor=BG, margin=dict(l=40,r=20,t=50,b=40))
                st.plotly_chart(fig, use_container_width=True)

    # ── TAB 5: State Space ──
    with tabs[4]:
        st.header("State Space")
        fig3d = px.scatter_3d(fpd, x="er_mean", y="ppi_mean", z="sentiment_avg",
                              color="platform" if "platform" in fpd.columns else None,
                              hover_name="actor_id", hover_data={"er_mean":":.2f","ppi_mean":":.2f","sentiment_avg":":.2f","afi_mean":":.4f"},
                              title="ER x PPI x Sentiment", labels={"er_mean":"ER (%)","ppi_mean":"PPI","sentiment_avg":"Sentiment"},
                              opacity=0.7)
        fig3d.update_traces(marker=dict(size=5, line=dict(width=0.3, color="rgba(0,0,0,0.2)")))
        fig3d.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig3d, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(cooks_plot(fpd), use_container_width=True)
        with c2:
            if len(fpd) >= 50:
                hfig = px.density_heatmap(fpd, x="sentiment_avg", y="er_mean", nbinsx=30, nbinsy=30, color_continuous_scale="Viridis", title="Sentiment x Engagement Density")
                hfig.update_layout(plot_bgcolor=BG, margin=dict(l=40,r=20,t=50,b=40))
                st.plotly_chart(hfig, use_container_width=True)

    # ── TAB 6: Actors ──
    with tabs[5]:
        st.header("Actor Explorer")
        actor_ids = filtered["actor_id"].to_list()
        selected = st.selectbox("Select Actor", actor_ids)
        if selected:
            actor = filtered.filter(pl.col("actor_id") == selected)
            if len(actor) > 0:
                r = actor.row(0, named=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Platform", r.get("platform","?"))
                c2.metric("Followers", f"{r.get('max_followers',0):,}")
                c3.metric("ER", f"{r.get('er_mean',0):.2f}%")
                c4.metric("PPI", f"{r.get('ppi_mean',0):.3f}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Sentiment", f"{r.get('sentiment_avg',0):.3f}")
                c2.metric("AFI", f"{r.get('afi_mean',0):.4f}")
                c3.metric("Ext. Ecosystem", "YES" if r.get("has_external_ecosystem") else "no")
                c4.metric("Prestige Drift", "YES" if r.get("prestige_drift_detected") else "no")

        st.markdown("---")
        st.plotly_chart(top_actors(filtered, 30), use_container_width=True)

    # ── TAB 7: Network ──
    with tabs[6]:
        st.header("Ecosystem Network")
        net = compute_network_metrics(filtered)
        cp = compute_cross_platform_report(filtered)

        if net.n_nodes == 0:
            st.info("Not enough data to build a network graph (minimum 3 nodes required).")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Nodes (Actors)", net.n_nodes)
            c2.metric("Edges (Connections)", net.n_edges)
            c3.metric("Network Density", f"{net.density:.5f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Degree", f"{net.avg_degree:.2f}")
            c2.metric("Clustering Coeff", f"{net.clustering_coefficient:.4f}")
            c3.metric("Modularity", f"{net.modularity:.4f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Communities", net.n_communities)
            c2.metric("Assortativity", f"{net.degree_assortativity:.4f}")
            c3.metric("Centralization", f"{net.centralization:.4f}")

            c1, c2, c3 = st.columns(3)
            c1.metric("K-Core (max)", net.k_core_number)
            c2.metric("Rich Club Coeff", f"{net.rich_club_coefficient:.4f}")
            c3.metric("Density", f"{net.density:.5f}")

            st.subheader("Centrality Rankings")
            c1, c2 = st.columns(2)
            with c1:
                if net.top_degree:
                    st.markdown("**Degree Centrality**")
                    st.dataframe(pd.DataFrame(net.top_degree, columns=["actor_id", "degree"]), use_container_width=True, hide_index=True)
                if net.top_pagerank:
                    st.markdown("**PageRank**")
                    st.dataframe(pd.DataFrame(net.top_pagerank, columns=["actor_id", "pagerank"]), use_container_width=True, hide_index=True)
            with c2:
                if net.top_betweenness:
                    st.markdown("**Betweenness**")
                    st.dataframe(pd.DataFrame(net.top_betweenness, columns=["actor_id", "betweenness"]), use_container_width=True, hide_index=True)
                if net.top_eigenvector:
                    st.markdown("**Eigenvector**")
                    st.dataframe(pd.DataFrame(net.top_eigenvector, columns=["actor_id", "eigenvector"]), use_container_width=True, hide_index=True)

            st.subheader("Cross-Platform Bridges")
            c1, c2 = st.columns(2)
            c1.metric("Total Bridges", cp["total_bridges"])
            c2.metric("Cross-Domain Bridges", cp["cross_domain_bridges"])
            if cp["top_pairs"]:
                pairs_df = pd.DataFrame(cp["top_pairs"])
                st.dataframe(pairs_df, use_container_width=True, hide_index=True)
            if cp["top_domain_pairs"]:
                st.markdown("**Cross-Domain Connections**")
                st.dataframe(pd.DataFrame(cp["top_domain_pairs"]), use_container_width=True, hide_index=True)

            if cp["top_bridges"]:
                with st.expander("Top Inter-Platform Connections"):
                    bridges_df = pd.DataFrame(cp["top_bridges"])
                    bridges_df["similarity"] = bridges_df["similarity"].round(4)
                    st.dataframe(bridges_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("Attention Observatory — Empirical model of digital attention distribution.")
    if only_attention:
        st.caption("HuggingFace excluded (corpus_dataset, not an attention network).")
    st.caption("Gini CI 95% via bootstrap (n=1000). Power Law α via MLE (Clauset 2009).")

if __name__ == "__main__":
    main()
