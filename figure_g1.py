import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from network_belief import NetworkBelief


def reproduce_figure_g1(
    n: int = 10, m: int = 10, o: int = 7,
    n_runs: int = 1000, steps: int = 100,
    beta_low: float = 0.5, beta_high: float = 2.0, beta_base: float = 1.0,
    focal_belief_idx: int = 0, n_traces: int = 5, trace_node: int = 0,
    seed: int | None = 0,
):
    """Simulate the β_pers, β_soc and β_ext sweeps of Dalege et al. (2024), Appendix G, and draw Figure G1 (b)–(n).

    Each sweep sets one β to `beta_low` or `beta_high` and keeps the other two at `beta_base`.
    Defaults follow the paper: 10 individuals in a fully connected external network, 10 personal
    beliefs each, 7 states, 100 iterations, 1,000 runs per condition (ω = .4, ρ = 1, α = 1.4 are the
    `NetworkBelief` defaults).

    Returns (fig, df). `df` is long format with columns
    sweep, beta, run, iteration, node, metric, value; `node` is NaN for run-level metrics:
      mean_belief        (run, node)       mean of the node's personal beliefs at the end of the run
      mean_belief_trace  (run, iteration, node)  same, at every iteration, only for the n_traces runs per β_pers
      final_focal        (run, node)       focal belief at the end of the run
      mean_social        (run, node)       mean of the node's social beliefs about others' focal beliefs
      social_var         (run, node)       variance of those social beliefs
      social_distance    (run)             mean |s_ij - b_j| over all i != j

    Choices the paper leaves open: social beliefs about oneself (j == i) are excluded; the distance in
    (m, n) is the mean absolute difference; a group has a positive majority when more than n/2 focal
    beliefs end > 0; traces in (d, e) show node `trace_node`.
    """
    f = focal_belief_idx
    others = ~np.eye(n, dtype=bool)
    seq = np.random.SeedSequence(seed)
    trace_rng = np.random.default_rng(seq.spawn(1)[0])
    columns = {c: [] for c in ("sweep", "beta", "run", "iteration", "node", "metric", "value")}

    def record(sweep, beta, run, metric, value, iteration=steps, node=np.nan):
        value = np.ravel(value)
        size = value.size
        columns["sweep"].append(np.full(size, sweep, dtype=object))
        columns["beta"].append(np.full(size, beta))
        columns["run"].append(np.full(size, run))
        columns["iteration"].append(np.broadcast_to(iteration, size))
        columns["node"].append(np.broadcast_to(node, size).astype(float))
        columns["metric"].append(np.full(size, metric, dtype=object))
        columns["value"].append(value)

    nodes = np.arange(n)
    for sweep in ("pers", "soc", "ext"):
        for beta in (beta_low, beta_high):
            betas = {"beta_pers": beta_base, "beta_soc": beta_base, "beta_ext": beta_base, f"beta_{sweep}": beta}
            traced = set(trace_rng.choice(n_runs, min(n_traces, n_runs), replace=False)) if sweep == "pers" else set()
            for run, run_seed in enumerate(seq.spawn(n_runs)):
                history = NetworkBelief(n, m, o, **betas, focal_belief_idx=f, seed=run_seed).run(steps)
                b, s = history.b[-1], history.s[-1]
                social = s[:, f, :][others].reshape(n, n - 1)
                actual = np.broadcast_to(b[None, :, f], (n, n))[others].reshape(n, n - 1)

                record(sweep, beta, run, "mean_belief", b.mean(axis=1), node=nodes)
                record(sweep, beta, run, "final_focal", b[:, f], node=nodes)
                record(sweep, beta, run, "mean_social", social.mean(axis=1), node=nodes)
                record(sweep, beta, run, "social_var", social.var(axis=1), node=nodes)
                record(sweep, beta, run, "social_distance", np.abs(social - actual).mean())
                if run in traced:
                    t, i = np.meshgrid(np.arange(steps + 1), nodes, indexing="ij")
                    record(sweep, beta, run, "mean_belief_trace", history.b.mean(axis=2), iteration=t.ravel(), node=i.ravel())

    df = pd.DataFrame({c: np.concatenate(v) for c, v in columns.items()})

    def get(sweep, beta, metric):
        return df[(df.sweep == sweep) & (df.beta == beta) & (df.metric == metric)]

    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    ax = dict(zip("bcdefghiklmn", axes.flat))
    states = np.linspace(-1, 1, o)
    label = {"pers": r"$\beta_{pers}$", "soc": r"$\beta_{soc}$", "ext": r"$\beta_{ext}$"}

    for low, high, sweep in (("b", "c", "pers"), ("d", "e", "pers"), ("f", "g", "soc"),
                             ("h", "i", "soc"), ("k", "l", "ext"), ("m", "n", "ext")):
        for letter, beta in ((low, beta_low), (high, beta_high)):
            a = ax[letter]
            a.set_title(f"({letter})  {label[sweep]} = {beta:g}", loc="left")

            if letter in "bc":
                a.hist(get(sweep, beta, "mean_belief").value, bins=20, range=(-1, 1), color="tab:blue", edgecolor="white")
                a.set_xlim(-1.05, 1.05)
                a.set_xlabel("Mean belief state")
                a.set_ylabel("Number of runs")
            elif letter in "de":
                trace = get(sweep, beta, "mean_belief_trace")
                for _, run_trace in trace[trace.node == trace_node].groupby("run"):
                    a.plot(run_trace.iteration, run_trace.value)
                a.set_ylim(-1.1, 1.1)
                a.set_xlabel("Iteration")
                a.set_ylabel("Mean belief state")
            elif letter in "fg":
                pairs = pd.concat([get(sweep, beta, "mean_social"), get(sweep, beta, "final_focal")])
                pairs = pairs.pivot_table(index=["run", "node"], columns="metric", values="value")
                a.scatter(pairs.mean_social, pairs.final_focal, s=6, color="k", alpha=0.05)
                slope, intercept = np.polyfit(pairs.mean_social, pairs.final_focal, 1)
                x = np.array([-1, 1])
                a.plot(x, slope * x + intercept, color="tab:blue", lw=2)
                a.set_xlim(-1.05, 1.05)
                a.set_ylim(-1.1, 1.1)
                a.set_xlabel("Mean end state of social beliefs")
                a.set_ylabel("End state of focal belief")
            elif letter in "hi":
                sns.violinplot(y=get(sweep, beta, "social_var").value, inner="box", ax=a)
                a.set_xlabel("Density")
                a.set_ylabel("Variance of social beliefs")
            elif letter in "kl":
                focal = get(sweep, beta, "final_focal")
                positive = (focal.value > 0).groupby(focal.run).sum() > n / 2
                values = focal[focal.run.isin(positive.index[positive])].value.to_numpy()
                counts = (values[:, None] == states[None, :]).sum(axis=0)
                a.bar(states, counts, width=0.8 * (states[1] - states[0]), color="tab:blue")
                a.set_xlabel("End state of focal beliefs")
                a.set_ylabel("Frequency")
            else:
                sns.violinplot(y=get(sweep, beta, "social_distance").value, inner="box", ax=a)
                a.set_xlabel("Density")
                a.set_ylabel("Distance social vs. actual beliefs")

        lo = min(ax[low].get_ylim()[0], ax[high].get_ylim()[0])
        hi = max(ax[low].get_ylim()[1], ax[high].get_ylim()[1])
        for letter in (low, high):
            ax[letter].set_ylim(lo, hi)

    fig.tight_layout()
    return fig, df
