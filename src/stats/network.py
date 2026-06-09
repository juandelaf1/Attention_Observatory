"""Network analysis: reply graphs, similarity networks, ecosystem topology.

Derived from Radical Pesimismo §II (asimetría topológica) and Inviabilidad §IV
(fragmentación / límite termodinámico del feed).

Implements:
  - Feature similarity graph (cosine on scaled feature vectors)
  - HN reply graph (comment → story)
  - Cross-platform bridge detection
  - PageRank, Eigenvector, Betweenness, K-Core, Rich Club
  - Community detection (greedy modularity)
"""

import numpy as np
import polars as pl
import networkx as nx
from pathlib import Path
from collections import defaultdict
from typing import NamedTuple


class NetworkReport(NamedTuple):
    n_nodes: int
    n_edges: int
    avg_degree: float
    density: float
    clustering_coefficient: float
    n_communities: int
    modularity: float
    degree_assortativity: float
    centralization: float
    top_degree: list[tuple[str, float]]
    top_betweenness: list[tuple[str, float]]
    top_pagerank: list[tuple[str, float]]
    top_eigenvector: list[tuple[str, float]]
    k_core_number: int
    rich_club_coefficient: float


def _cosine_similarity(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sim = X @ X.T / (norms @ norms.T)
    return np.clip(sim, -1, 1)


def _cosine_network(df: pl.DataFrame, feature_cols: list[str] | None = None,
                    threshold: float = 0.85) -> nx.Graph:
    if feature_cols is None:
        feature_cols = ["er_mean", "ppi_mean", "sentiment_avg", "afi_mean"]
    pdf = df.select(["actor_id", "platform", "domain"] + feature_cols).to_pandas()
    pdf = pdf.dropna(subset=feature_cols)
    if len(pdf) < 2:
        return nx.Graph()
    ids = pdf["actor_id"].values
    platforms = pdf["platform"].values
    domains = pdf["domain"].values
    X = pdf[feature_cols].values.astype(np.float64)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    sim = _cosine_similarity(X)
    G = nx.Graph()
    for i in range(len(ids)):
        G.add_node(ids[i], platform=platforms[i], domain=domains[i])
    edges = np.argwhere(sim > threshold)
    edges = edges[edges[:, 0] < edges[:, 1]]
    for i, j in edges:
        G.add_edge(ids[i], ids[j], weight=float(sim[i, j]))
    return G


def _hn_reply_network(bronze_dir: str = "data/bronze") -> nx.DiGraph:
    hn_posts_path = Path(bronze_dir) / "hn_posts.parquet"
    if not hn_posts_path.exists():
        return nx.DiGraph()
    posts = pl.read_parquet(hn_posts_path)
    G = nx.DiGraph()
    for row in posts.iter_rows(named=True):
        actor = row.get("actor_id", "")
        story = str(row.get("story_id", ""))
        post_id = str(row.get("post_id", ""))
        if actor:
            G.add_node(actor, type="actor")
        if story and story != post_id:
            story_node = f"story_{story}"
            G.add_node(story_node, type="story")
            G.add_edge(actor, story_node, weight=1.0)
    return G


def _cross_platform_bridges(G_cos: nx.Graph) -> list[dict]:
    bridges = []
    for u, v in G_cos.edges():
        pu = G_cos.nodes[u].get("platform", "")
        pv = G_cos.nodes[v].get("platform", "")
        du = G_cos.nodes[u].get("domain", "")
        dv = G_cos.nodes[v].get("domain", "")
        if pu and pv and pu != pv:
            bridges.append({
                "source": u, "target": v,
                "platform_a": pu, "platform_b": pv,
                "domain_a": du, "domain_b": dv,
                "similarity": G_cos[u][v].get("weight", 0)
            })
    bridges = sorted(bridges, key=lambda x: x["similarity"], reverse=True)
    return bridges


def _compute_rich_club(G: nx.Graph, k: int = 5) -> float:
    try:
        core = nx.k_core(G, k=k)
        if core.number_of_nodes() < 2 or core.number_of_edges() < 1:
            return 0.0
        observed_density = nx.density(core)
        n = core.number_of_nodes()
        m_complete = n * (n - 1) / 2
        random_density = core.number_of_edges() / m_complete if m_complete > 0 else 0
        return float(observed_density / max(0.001, random_density))
    except Exception:
        return 0.0


def compute_network_metrics(df: pl.DataFrame, max_nodes: int = 500) -> NetworkReport:
    if len(df) > max_nodes:
        df = df.sort("er_mean", descending=True).head(max_nodes)

    G_cos = _cosine_network(df)
    G_di = _hn_reply_network()

    G_combined = nx.Graph()
    for n, d in G_cos.nodes(data=True):
        G_combined.add_node(n, **d)
    for n, d in G_di.nodes(data=True):
        if n not in G_combined:
            G_combined.add_node(n, **d)
    for u, v, d in G_cos.edges(data=True):
        G_combined.add_edge(u, v, **d)
    for u, v in G_di.edges():
        if not G_combined.has_edge(u, v):
            G_combined.add_edge(u, v, weight=0.5)

    if G_combined.number_of_nodes() < 3:
        return NetworkReport(0, 0, 0, 0, 0, 0, 0, 0, 0, [], [], [], [], 0, 0)

    from networkx.algorithms.community import greedy_modularity_communities
    deg = dict(G_combined.degree())
    n_nodes = G_combined.number_of_nodes()
    n_edges = G_combined.number_of_edges()
    avg_deg = float(np.mean(list(deg.values()))) if deg else 0
    density = nx.density(G_combined)
    cc = nx.average_clustering(G_combined)
    try:
        comms = list(greedy_modularity_communities(G_combined))
        mod = nx.community.modularity(G_combined, comms)
    except Exception:
        comms = []
        mod = 0.0
    try:
        assort = nx.degree_assortativity_coefficient(G_combined)
    except Exception:
        assort = 0.0
    try:
        cent = nx.degree_centrality(G_combined)
        cent_max = max(cent.values()) if cent else 0
        cent_sum = sum(cent.values()) if cent else 0
        centralization = (cent_sum / max(1, n_nodes - 1)) / max(1, n_nodes)
    except Exception:
        centralization = 0.0

    top_deg = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:10]
    try:
        bc = nx.betweenness_centrality(G_combined, k=min(200, n_nodes), normalized=True)
        top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]
    except Exception:
        top_bc = []
    try:
        pr = nx.pagerank(G_combined, weight="weight")
        top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:10]
    except Exception:
        top_pr = []
    try:
        ev = nx.eigenvector_centrality(G_combined, max_iter=1000, tol=1e-6)
        top_ev = sorted(ev.items(), key=lambda x: x[1], reverse=True)[:10]
    except Exception:
        top_ev = []

    try:
        k_core = nx.core_number(G_combined)
        k_core_val = max(k_core.values()) if k_core else 0
    except Exception:
        k_core_val = 0

    rich_club = _compute_rich_club(G_combined)

    return NetworkReport(
        n_nodes=n_nodes,
        n_edges=n_edges,
        avg_degree=avg_deg,
        density=density,
        clustering_coefficient=cc,
        n_communities=len(comms),
        modularity=mod,
        degree_assortativity=assort,
        centralization=centralization,
        top_degree=[(str(k), float(v)) for k, v in top_deg],
        top_betweenness=[(str(k), float(v)) for k, v in top_bc],
        top_pagerank=[(str(k), float(v)) for k, v in top_pr],
        top_eigenvector=[(str(k), float(v)) for k, v in top_ev],
        k_core_number=k_core_val,
        rich_club_coefficient=rich_club,
    )


def compute_cross_platform_report(df: pl.DataFrame, max_nodes: int = 500) -> dict:
    if len(df) > max_nodes:
        df = df.sort("er_mean", descending=True).head(max_nodes)
    G_cos = _cosine_network(df)
    bridges = _cross_platform_bridges(G_cos)

    platform_pairs = defaultdict(lambda: {"count": 0, "total_sim": 0.0})
    for b in bridges:
        key = tuple(sorted([b["platform_a"], b["platform_b"]]))
        platform_pairs[key]["count"] += 1
        platform_pairs[key]["total_sim"] += b["similarity"]

    pairs_summary = sorted(
        [{"pair": f"{k[0]}-{k[1]}", "count": v["count"],
          "mean_sim": v["total_sim"] / max(1, v["count"])}
         for k, v in platform_pairs.items()],
        key=lambda x: x["count"], reverse=True
    )

    domain_pairs = defaultdict(lambda: {"count": 0})
    for b in bridges:
        da, db = b["domain_a"], b["domain_b"]
        if da and db and da != db:
            key = tuple(sorted([da, db]))
            domain_pairs[key]["count"] += 1

    domain_summary = sorted(
        [{"pair": f"{k[0]}-{k[1]}", "count": v["count"]}
         for k, v in domain_pairs.items()],
        key=lambda x: x["count"], reverse=True
    )

    return {
        "total_bridges": len(bridges),
        "unique_pairs": len(pairs_summary),
        "top_pairs": pairs_summary[:10],
        "top_bridges": bridges[:20],
        "cross_domain_bridges": len(domain_pairs),
        "top_domain_pairs": domain_summary[:10],
    }
