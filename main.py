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
from numpy.typing import ArrayLike, NDArray
from scratch import B_states, node_indx

def _to_array(name: str, value: ArrayLike) -> NDArray:
    """Convert value to an ndarray, naming the argument if conversion fails."""
    try:
        return np.asarray(value)
    except ValueError as e:
        raise ValueError(f"{name} could not be converted to an array: {e}") from e

def _to_float(name: str, value: float) -> float:
    """Convert value to a float, naming the argument if conversion fails."""
    try:
        return float(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"{name} could not be converted to a float: {e}") from e

def _check_shape(A: NDArray, expected: tuple, name):
    if A.shape != tuple(expected):
        raise ValueError(f"{name}, wrong shape {A.shape} != {expected}")

def _check_int(name: str, value: int) -> int:
    if not isinstance(value,int): raise ValueError(f"{name} must be an integer")

class NetworkBelief:

    def __init__(
        self,
        s: int,
        B: ArrayLike,
        S: ArrayLike,
        W_pers: ArrayLike,
        W_soc: ArrayLike,
        W_ext: ArrayLike,
        beta_pers: float,
        beta_soc: float,
        beta_ext: float,
    ) -> None:
        
        _check_int("s", s)
        self.s = s
        self.B = _to_array("B", B)
        self.S = _to_array("S", S)
        self.W_pers = _to_array("W_pers", W_pers)
        self.W_soc = _to_array("W_soc", W_soc)
        self.W_ext = _to_array("W_ext", W_ext)


        if self.B.ndim != 2:
            raise ValueError(f"B, wrong number of dimensions {self.B.ndim} != 2")
        
        self.n, self.b = self.B.shape
        _check_shape(self.S, (self.n, self.b, self.n), "S")
        _check_shape(self.W_pers, (self.n, self.b, self.b), "W_pers")
        _check_shape(self.W_soc, (self.n, self.b, self.n), "W_soc")
        _check_shape(self.W_ext, (self.n, self.b, self.n), "W_ext")

        states = np.linspace(-1, 1, s)
        self.B_states = np.tile(states, (self.n, self.b, 1))
        self.S_states = np.tile(states, (self.n, self.b, self.n, 1))

        self.beta_pers = _to_float("beta_pers", beta_pers)
        self.beta_soc = _to_float("beta_soc", beta_soc)
        self.beta_ext = _to_float("beta_ext", beta_ext)


    def get_delta_h_pers(self) -> NDArray:
        return (self.T + np.einsum('nijs,nis->njs', self.W_pers, self.B))*(self.B - self.B_states[:, :])
    
    def get_delta_b_h_soc(self) -> NDArray:
        return np.einsum('nijs,njis->nis', self.W_soc, self.S)*(self.B - self.B_states[:,:])
    
    def get_delta_s_h_soc(self) -> NDArray:
        return np.einsum('nijs,njis->nis', self.W_soc, self.S)*(self.B - self.B_states[:,:])
    
    
    def get_h_ext(self) -> NDArray:
        return  np.einsum('nijs,njis->nis', self.W_ext, self.S)*(self.B - self.S)
        
    def get_D_pers(self) -> float:
        return self.beta_pers*self.get_h_pers(node_indx, B) + self.beta_soc*self.get_h_soc(node_indx, B)
    
    def get_D_soc(self, node_indx: int, B: np.array) -> float:
        return self.beta_soc*self.get_h_soc(node_indx, B) + self.beta_ext*self.get_h_ext(node_indx, B)
    

    



