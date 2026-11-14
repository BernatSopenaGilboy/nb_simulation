import numpy as np
import pytest
from network_belief import NetworkBelief

N, M, O = 3, 2, 3  # candidate states v = [-1, 0, 1]


@pytest.fixture
def model():
    m = NetworkBelief(N, M, O, beta_pers=1.0, beta_soc=1.0, beta_ext=1.0, seed=0)

    # beliefs b[agent, belief]
    m.b = np.array([[1.0, -1.0],
                    [0.0, 1.0],
                    [-1.0, 1.0]])
    # external field acts the same on all agents: each column is constant
    m.tau = np.array([[1.0, 0.0],
                      [1.0, 0.0],
                      [1.0, 0.0]])
    m.w = np.tile([[0.0, 0.5],
                   [0.5, 0.0]], (N, 1, 1))

    # social spins s[agent, belief, :]
    m.s = np.array([[[1, 1, 0], [1, -1, -1]],
                    [[1, 0, 0], [1, 1, 1]],
                    [[-1, -1, 0], [1, 1, 0]]], dtype=float)
    m.rho = np.ones((N, M, N))
    m.rho[:, 1, :] = 0.5
    return m


# Personal field = tau + w . b, delta = -field * (v - b_ij):
#   (0,0): field  0.5 -> [ 1.0,  0.5,  0.0]
#   (0,1): field  0.5 -> [ 0.0, -0.5, -1.0]
#   (1,0): field  1.5 -> [ 1.5,  0.0, -1.5]
#   (1,1): field  0.0 -> [ 0.0,  0.0,  0.0]
#   (2,0): field  1.5 -> [ 0.0, -1.5, -3.0]
#   (2,1): field -0.5 -> [-1.0, -0.5,  0.0]
#
# Social field = rho . s, delta = -field * (v - b_ij):
#   (0,0): field  2.0 -> [ 4.0,  2.0,  0.0]
#   (0,1): field -0.5 -> [ 0.0,  0.5,  1.0]
#   (1,0): field  1.0 -> [ 1.0,  0.0, -1.0]
#   (1,1): field  1.5 -> [ 3.0,  1.5,  0.0]
#   (2,0): field -2.0 -> [ 0.0,  2.0,  4.0]
#   (2,1): field  1.0 -> [ 2.0,  1.0,  0.0]
#
# `belief` is always an int index; `agent` is an int or a list of ints.
# Both methods return (len(agent), 1, O), with len(agent) = 1 for an int.


# ---------- delta_H_pers_b ----------

def test_delta_H_pers_b_single_agent(model):
    result = model.delta_H_pers_b(1, 0)  # (1,0)
    np.testing.assert_allclose(result, [[[1.5, 0.0, -1.5]]])


def test_delta_H_pers_b_list_of_agents(model):
    result = model.delta_H_pers_b([0, 2], 1)  # (0,1), (2,1)
    np.testing.assert_allclose(result, [[[0.0, -0.5, -1.0]],
                                        [[-1.0, -0.5, 0.0]]])


# ---------- delta_H_soc_b ----------

def test_delta_H_soc_b_single_agent(model):
    result = model.delta_H_soc_b(1, 0)  # (1,0)
    np.testing.assert_allclose(result, [[[1.0, 0.0, -1.0]]])


def test_delta_H_soc_b_list_of_agents(model):
    result = model.delta_H_soc_b([0, 2], 1)  # (0,1), (2,1)
    np.testing.assert_allclose(result, [[[0.0, 0.5, 1.0]],
                                        [[2.0, 1.0, 0.0]]])


# ---------- shapes and validity ----------

@pytest.mark.parametrize("agent, k", [(1, 1), ([0, 2], 2), (np.arange(N), N)])
def test_delta_D_pers_shape(model, agent, k):
    assert model.delta_D_pers(agent, 0).shape == (k, 1, O)


@pytest.mark.parametrize("method", ["delta_H_soc_s", "delta_H_ext_s", "delta_D_soc"])
@pytest.mark.parametrize("agent, k", [(1, 1), ([0, 2], 2), (np.arange(N), N)])
def test_social_deltas_shape(model, method, agent, k):
    assert getattr(model, method)(agent, 1).shape == (k, N, O)


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
