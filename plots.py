import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_node_beliefs(history: NDArray, node: int, ax=None):
    """Plot every belief of one node over time from a (steps + 1, n, m) history."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    t = np.arange(history.shape[0])
    for k in range(history.shape[2]):
        ax.step(t, history[:, node, k], where="post", label=f"topic {k}")

    ax.set_xlabel("step")
    ax.set_ylabel("belief")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"Beliefs of node {node}")
    ax.legend(loc="upper right")
    return ax


def plot_mean_beliefs(history: NDArray, ax=None):
    """Plot the mean belief across nodes for each topic over time, with a ±1 std band."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    t = np.arange(history.shape[0])
    mean, std = history.mean(axis=1), history.std(axis=1)  # (steps + 1, m)
    for k in range(history.shape[2]):
        line, = ax.plot(t, mean[:, k], label=f"topic {k}")
        ax.fill_between(t, mean[:, k] - std[:, k], mean[:, k] + std[:, k],
                        color=line.get_color(), alpha=0.15)

    ax.set_xlabel("step")
    ax.set_ylabel("mean belief")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("Mean belief per topic (±1 std across nodes)")
    ax.legend(loc="upper right", fontsize="small", ncol=2)
    return ax


def plot_belief_heatmaps(history: NDArray, topics=None):
    """Plot a nodes × time heatmap of beliefs for each selected topic."""
    topics = range(history.shape[2]) if topics is None else topics
    fig, axes = plt.subplots(len(topics), 1, figsize=(8, 1.8 * len(topics)),
                             sharex=True, squeeze=False)

    for ax, k in zip(axes[:, 0], topics):
        im = ax.imshow(history[:, :, k].T, aspect="auto", cmap="coolwarm",
                       vmin=-1, vmax=1, interpolation="nearest")
        ax.set_ylabel(f"topic {k}\nnode")
    axes[-1, 0].set_xlabel("step")
    fig.colorbar(im, ax=axes[:, 0], label="belief")
    return fig


def plot_belief_histograms(history: NDArray, v: NDArray, t: int = -1):
    """Plot the distribution of beliefs across nodes at step t, one panel per topic."""
    m = history.shape[2]
    cols = min(m, 5)
    rows = int(np.ceil(m / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 1.8 * rows),
                             sharex=True, sharey=True, squeeze=False)

    width = 0.8 * (v[1] - v[0]) if len(v) > 1 else 0.5
    for k, ax in enumerate(axes.flat):
        if k >= m:
            ax.axis("off")
            continue
        counts = (history[t, :, k][:, None] == v[None, :]).sum(axis=0)
        ax.bar(v, counts, width=width)
        ax.set_title(f"topic {k}", fontsize="small")
    for ax in axes[-1]:
        ax.set_xlabel("belief")
    for ax in axes[:, 0]:
        ax.set_ylabel("nodes")
    fig.suptitle(f"Belief distribution at step {t % history.shape[0]}")
    fig.tight_layout()
    return fig


def plot_topic_correlation(history: NDArray, last: int = 1, ax=None):
    """Plot the topic × topic correlation of beliefs across nodes, pooled over the last `last` steps."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))

    samples = history[-last:].reshape(-1, history.shape[2])  # (last * n, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(samples, rowvar=False)  # NaN for topics with no variation (full consensus)

    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xlabel("topic")
    ax.set_ylabel("topic")
    ax.set_title(f"Topic correlation (last {last} steps)")
    plt.colorbar(im, ax=ax, label="correlation")
    return ax
