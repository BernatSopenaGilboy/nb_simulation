from collections import deque
from network_belief import BaseNetworkBelief

# Model:
# There is one focal belief, the agents have access to the social belief of their contacts
# and also their network structure however their beliefs values remain private
#
# Belief update: dissonance with
#   Internal = - sum_ij w_ij x_i x_j
#   Social   = - sum_k rho_ik x_i^foc x_k^foc
#   H = beta_pers * internal + beta_soc * social
#
# Edge-weight update (Hebbian + social adaptation, Eq. 3):
#   dx_m = (x_m(t) - x_m(t - tau)) / tau
#   w_mn <- w_mn + eps * dx_m * dx_n + mu * (mean_{k in K(A)} w_mn^(k) - w_mn)

import numpy as np
from numpy.typing import NDArray


class ANB(BaseNetworkBelief):
    def __init__(
        self,
        n: int,
        m: int,
        o: int,
        beta_pers: float,
        beta_soc: float,
        focal_belief_idx: int,
        memory: int = 1,
        epsilon: float = 1.0,
        mu: float = 1.0,
        strength_external_field: float = 0.0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        """
        Parameters
        ----------
        n : int
            Number of agents (nodes).
        m : int
            Number of beliefs per agent.
        o : int
            Number of candidate states per spin, evenly spaced in [-1, 1].
        beta_pers : float
            Weight of the personal energy H_pers in the belief dissonance.
        beta_soc : float
            Weight of the social energy H_soc in the belief dissonance.
        focal_belief_idx : int
            Belief index that is socially coupled via rho.
        memory : int
            Activation memory tau: lag length for dx = (b(t) - b(t-tau)) / tau.
        epsilon : float
            Strength of internal (Hebbian) adaptation of w.
        mu : float
            Strength of social adaptation of w toward contacts' mean w.
        strength_external_field : float
            Unused for now (external field / tau left at 0).
        seed : int, numpy.random.Generator or None, optional
            Seed or generator for the initial spins and the dynamics.
        """
        super().__init__(n, m, o, beta_pers, beta_soc, 0.0, focal_belief_idx, seed)

        self.beta_ext = None
        self.s = None  # no social perception spins in this model
        self.tau = 0  # external field off
        self.memory = int(memory)
        self.epsilon = float(epsilon)
        self.mu = float(mu)
        # Past beliefs only, oldest -> newest; maxlen=tau auto-drops oldest on append
        self.b_mem: deque[NDArray] = deque(
            (self.b.copy() for _ in range(self.memory)),
            maxlen=self.memory,
        )

    def dx_beliefs(self) -> NDArray:
        """Telescoped activation change (n, m): (b(t) - b(t - tau)) / tau."""
        return (self.b - self.b_mem[0]) / self.memory

    def update_w(self, agent: int, dx: NDArray) -> None:
        """Hebbian + social adaptation of one agent's belief network w[agent]."""
        hebbian = self.epsilon * np.outer(dx, dx)  # (m, m)

        rho_a = self.rho[agent, self.focal_belief_idx, :]  # (n,); self already 0
        w_bar = (rho_a @ self.w.reshape(self.n, -1)).reshape(self.m, self.m) / rho_a.sum()
        social = self.mu * (w_bar - self.w[agent])

        self.w[agent] += hebbian + social
        self.w[agent, np.arange(self.m), np.arange(self.m)] = 0

    def delta_D(self, agent: int, belief: int) -> NDArray:  # (o,)
        """Change in felt dissonance for moving b[agent, belief] to each candidate state."""
        internal = self.w[agent, belief] @ self.b[agent]
        social = self.rho[agent, belief] @ self.b[:, belief]
        felt_field = self.beta_pers * internal + self.beta_soc * social
        return -felt_field * (self.v - self.b[agent, belief])

    def run_step(self) -> None:
        # Belief update
        belief_order = self.rng.permuted(np.tile(np.arange(self.m), (self.n, 1)))
        for i in self.rng.permutation(self.n):
            for k in belief_order[i, :]:
                probs_b = self.glauber_prob(-self.delta_D(i, k))
                self.b[i, k] = self.sample(probs_b)
            # weight update
            dx = self.dx_beliefs()
            self.update_w(i, dx[i])
     

        self.b_mem.append(self.b.copy())
