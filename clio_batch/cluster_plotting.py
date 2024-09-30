"""Various plotting functions to look at results in 2D."""

import matplotlib.pyplot as plt
import distinctipy
import networkx as nx
import matplotlib.patches
import numpy as np


def plot_requery_weights(ws, picker):
    """Plot requery weight values."""
    fig, ax = plt.subplots()
    weights = [x[1] for x in ws.history]
    ax.plot(weights)
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Utility")
    plt.show()


def plot_ib_weights(ws, picker):
    """Plot IB weight values."""
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(picker.deltas)
    ax[0].set_ylabel("delta(m)")
    ax[1].plot(picker.betas)
    ax[1].set_ylabel("Beta")
    ax[0].set_xlabel("Iterations")
    plt.show()


def draw_clusters(
    ax,
    G,
    ws,
    edge_color=(0.22, 0.22, 0.22, 0.4),
    node_size=40,
    edge_width=0.5,
    pastel_factor=0.7,
    **kwargs,
):
    """Draw resulting clusters."""
    pos = {x: G.nodes[x]["position"][:2] for x in G}
    # make colormap indexed by cluster id
    cmap = distinctipy.get_colors(ws.num_clusters, pastel_factor=pastel_factor)
    cmap_lookup = {x: cmap[i] for i, x in ws.cluster_ids.items()}
    # assign colors by looking up cluster id for each node
    cluster_map = ws.cluster_map
    colors = [cmap_lookup[cluster_map[x]] for x in G]
    # draw the graph
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width, ax=ax)
    nx.draw_networkx_nodes(
        G, pos, node_color=colors, node_size=node_size, ax=ax, **kwargs
    )
    ax.set_aspect("equal")


def draw_task_clusters(
    ax,
    G,
    ws,
    score,
    edge_color=(0.22, 0.22, 0.22, 0.4),
    node_size=40,
    edge_width=0.5,
    pastel_factor=0.7,
    **kwargs,
):
    """Draw clusters colored by nearest task."""
    pos = {x: G.nodes[x]["position"][:2] for x in G}
    cmap = distinctipy.get_colors(len(score.tasks), pastel_factor=pastel_factor)

    cluster_features = np.array(ws.get_merged_features(G, f_score=score))
    best_tasks = score.get_best_tasks(cluster_features)

    cmap_lookup = {x: cmap[best_tasks[i]] for i, x in ws.cluster_ids.items()}
    colors = [cmap_lookup[ws.clusters[ws.order[x]]] for x in G]
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_size, ax=ax)

    patches = [matplotlib.patches.Patch(edgecolor="k", facecolor=c) for c in cmap]
    ax.legend(
        patches, score.tasks, loc="lower left", ncol=3, bbox_to_anchor=(-0.5, -0.15)
    )
    ax.set_aspect("equal")


def draw_best_tasks(
    ax,
    G,
    ws,
    score,
    edge_color=(0.22, 0.22, 0.22, 0.4),
    node_size=40,
    edge_width=0.5,
    pastel_factor=0.7,
    **kwargs,
):
    """Draw input graph colored by best task."""
    pos = {x: G.nodes[x]["position"][:2] for x in G}
    cmap = distinctipy.get_colors(len(score.tasks), pastel_factor=pastel_factor)

    X = np.array([G.nodes[x]["semantic_feature"] for x in G])
    best_tasks = score.get_best_tasks(X)
    colors = [cmap[x] for x in best_tasks]
    nx.draw_networkx_edges(G, pos, edge_color=edge_color, width=edge_width, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=node_size, ax=ax)

    patches = [matplotlib.patches.Patch(edgecolor="k", facecolor=c) for c in cmap]
    ax.legend(
        patches, score.tasks, loc="lower left", ncol=3, bbox_to_anchor=(-0.5, -0.15)
    )
    ax.set_aspect("equal")
