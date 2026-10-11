#Things we need to save

# Spins of the network internal B (n,b) social S (n, n, b)
# Weights matrices, W_persernal, W_external

# W_persernal: w_ij (b, b), W_social (RHo) W_external: (n, n) (rows social nodes, cols focal belief nodes), 
# W_field, t_i (1, b) 

# Update rules:

# beliefs update from state i to j: p(i->j) = 1/(1+e\delta(i,j) D) / \sum_k (1 / (1 +e\delta(i,k) D))

# Where D is the felt dissonance, we ahve three types

#H_persernal = - \sum_i \tau_i b_i - \sum_ij wij bi (j\neqi)-> W internal (n, b, b) (tau in the diagonal)
#H_soc = - \sum_k \sum_i \rho_ik b_i s_ik -> \Rho W social (n, b, n)
#H_ext = -\sum_i \sum_k \alpha_k s_ik b_ik -> \alpha_jik (n, b, n)

#D_pers = \beta_pers H_pers + \beta_soc H_soc
#D_soc = \beta_soc H_soc + \beta_ext H_ext  ## ALL BELIEFS HAVE A SOCIAL BELIEF

import numpy as np
from numpy.typing import  NDArray
from scipy.special import expit


def _to_float(name: str, value: float) -> float:
    """Convert value to a float, naming the argument if conversion fails."""
    try:
        return float(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{name} could not be converted to a float: {e}") from e

class NetworkBelief:
    tau_strength = 0
    w_strength = 0.4
    rho_strength = 1
    alpha_strength = 1.4

    def __init__(
        self, n, m, o, 
        
        beta_pers, beta_soc, beta_ext,
        seed=None
    ) -> None:
    
        self.rng = np.random.default_rng(seed)

        self.w = np.ones((n, m, m))*self.w_strength # Coupling internal belief
        self.w[:, np.arange(m), np.arange(m)] = 0 #Diag to 0

        self.rho = np.ones((n, m, n))*self.rho_strength # Coupling internal-social beliefs
        self.alpha = np.ones((n, m, n))*self.alpha_strength # Coupling external-social beliefs
        self.tau = np.ones((n,m))*self.tau_strength

        self.v = np.linspace(-1,1,o) # Candidate states
        self.b = self.rng.choice(self.v, size=(n,m)) # belief spins
        self.s = self.rng.choice(self.v, (n,m,n)) # social spins


        self.beta_pers = _to_float("beta_pers", beta_pers)
        self.beta_soc = _to_float("beta_soc", beta_soc)
        self.beta_ext = _to_float("beta_ext", beta_ext)


    # delta = H_current - H_candidate

    def delta_H_pers_b(self) -> NDArray: #(n, m, o)
        F_pers = self.tau + np.einsum('ijk,ij->ik', self.w, self.b) #Local pers field
        H_b = - F_pers*self.b
        H_pers_candidates = - F_pers[:, :, None] * self.v[None, None, :]
        return H_b[:, :, None] - H_pers_candidates

    def delta_H_soc_b(self) -> NDArray: #(n, m, o)
        F_soc = - np.einsum('jik,jik->ji', self.rho, self.s)        # (J, I): social field on each personal belief
        H_b = F_soc*self.b
        H_soc_candidates = F_soc[:, :, None]* self.v[None, None, :]     # (J, I, O): energy of every candidate state
        return H_b[:, :, None] - H_soc_candidates

    def delta_H_soc_s(self) -> NDArray: #(n, m, n, o)
        F_soc = - self.rho*self.b[:, :, None]
        H_s = F_soc*self.s
        H_soc_candidates = F_soc[:, :, :, None]*self.v[None, None, None, :]
        return H_s[:, :, :, None] - H_soc_candidates

    def delta_H_ext_s(self) -> NDArray: #(n, m, n, o)
        F_ext = - self.alpha*self.b.T[None, :, :]
        H_s = F_ext * self.s
        H_ext_candidates = F_ext[:, :, :, None] * self.v
        return H_s[:, :, :, None] - H_ext_candidates

    def get_delta_D_pers(self) -> NDArray: #(n, m, o)
        return self.beta_pers*self.delta_H_pers_b() + self.beta_soc*self.delta_H_soc_b()

    def get_delta_D_soc(self) -> NDArray: #(n, m, n, o)
        return self.beta_soc*self.delta_H_soc_s() + self.beta_ext*self.delta_H_ext_s()


    def glauber_prob(self, delta_D: NDArray):
        p = expit(delta_D)
        return p / p.sum(axis=-1, keepdims=True)

    def sample(self, p):

        # CDF trick
        size = p.shape[:-1] + (1,)
        u = self.rng.uniform(size = size) 
        cumsum = p.cumsum(axis=-1)
        indx = (cumsum < u).sum(axis=-1)
        return self.v[indx] #fancy indexing
    
    def run_step(self):

        delta_D_pers = self.get_delta_D_pers()
        delta_D_soc = self.get_delta_D_soc()

        self.b = self.sample(self.glauber_prob(delta_D_pers))
        self.s = self.sample(self.glauber_prob(delta_D_soc))
    
    def run(self, steps) -> NDArray: #(steps + 1, n, m)
        history = np.empty((steps + 1,) + self.b.shape)
        history[0] = self.b
        for t in range(1, steps + 1):
            self.run_step()
            history[t] = self.b
        return history




