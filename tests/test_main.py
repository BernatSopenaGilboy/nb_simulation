import numpy as np
import pytest
from network_belief import NetworkBelief

N, M, O = 3, 2, 4


@pytest.fixture
def model():
    return NetworkBelief(N, M, O, beta_pers=1.0, beta_soc=1.0, beta_ext=1.0, seed=0)


def test_init_shapes(model):
    assert model.b.shape == (N, M)
    assert model.s.shape == (N, M, N)
    assert np.all(np.diag(model.w[0]) == 0)


@pytest.mark.parametrize("method", ["delta_H_pers_b", "delta_H_soc_b"])
def test_belief_deltas_shape(model, method):
    assert getattr(model, method)().shape == (N, M, O)


@pytest.mark.parametrize("method", ["delta_H_soc_s", "delta_H_ext_s"])
def test_social_deltas_shape(model, method):
    assert getattr(model, method)().shape == (N, M, N, O)


def test_delta_D_shapes(model):
    assert model.get_delta_D_pers().shape == (N, M, O)
    assert model.get_delta_D_soc().shape == (N, M, N, O)


def test_glauber_prob_normalised(model):
    p = model.glauber_prob(model.get_delta_D_soc())
    assert np.all(p >= 0)
    assert np.allclose(p.sum(axis=-1), 1)


def test_run_step_keeps_valid_states(model):
    model.run_step()
    assert model.b.shape == (N, M)
    assert model.s.shape == (N, M, N)
    assert np.isin(model.b, model.v).all()
    assert np.isin(model.s, model.v).all()


def test_run_history_shape(model):
    history = model.run(5)
    assert history.b.shape == (6, N, M)
    assert history.s.shape == (6, N, M, N)
    assert np.array_equal(history.b[-1], model.b)
    assert np.array_equal(history.s[-1], model.s)


def test_sample_returns_valid_states(model):
    p = model.glauber_prob(model.get_delta_D_pers())
    x = model.sample(p)
    assert x.shape == (N, M)
    assert np.isin(x, model.v).all()
