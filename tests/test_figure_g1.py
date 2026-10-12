import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from figure_g1 import reproduce_figure_g1

N, M, O, RUNS, STEPS = 4, 3, 7, 3, 5


@pytest.fixture(scope="module")
def result():
    return reproduce_figure_g1(n=N, m=M, o=O, n_runs=RUNS, steps=STEPS, n_traces=2, seed=1)


def test_figure_has_twelve_panels(result):
    fig, _ = result
    assert len(fig.axes) == 12


def test_all_metrics_recorded(result):
    _, df = result
    assert set(df.metric) == {"mean_belief", "mean_belief_trace", "final_focal",
                              "mean_social", "social_var", "social_distance"}
    assert set(df.sweep) == {"pers", "soc", "ext"}


def test_focal_beliefs_are_discrete_states(result):
    _, df = result
    focal = df[df.metric == "final_focal"].value
    assert np.isin(focal, np.linspace(-1, 1, O)).all()


def test_row_counts(result):
    _, df = result
    per_node = df[df.metric == "social_var"]
    assert len(per_node) == 3 * 2 * RUNS * N
    assert len(df[df.metric == "social_distance"]) == 3 * 2 * RUNS
    assert len(df[df.metric == "mean_belief_trace"]) == 2 * 2 * (STEPS + 1) * N


def test_reproducible():
    _, a = reproduce_figure_g1(n=N, m=M, o=O, n_runs=2, steps=3, seed=7)
    _, b = reproduce_figure_g1(n=N, m=M, o=O, n_runs=2, steps=3, seed=7)
    pd.testing.assert_frame_equal(a, b)
