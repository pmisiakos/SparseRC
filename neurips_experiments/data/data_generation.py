import numpy as np
import igraph as ig
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import spsolve

def simulate_dag(d, s0, graph_type):
    "Adapted from NOTEARS repo"

    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    "Adapted from NOTEARS repo"

    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def sparse_rc_sem(W, n, sparsity=0.3, std=0.01, noise_type='gauss', noise_effect='both', fix_sup='False'):
    """
        Linear SEM occuring from sparse root causes
    """
    #number of nodes
    d = W.shape[0]
    I = np.eye(d)

    # initializing the sparse root_causes
    if(fix_sup == 'True'):
        pos = np.random.choice([0, 1], size=d, p=[1 - sparsity, sparsity])
    else: 
        pos = np.random.choice([0, 1], size=(n, d), p=[1 - sparsity, sparsity]) 
    C = pos * np.random.uniform(0, 1, size=(n, d)) 

    # computing matrix of independent noises
    if std==0:
        Ns = np.zeros((n, d))
        Nf = np.zeros((n, d))
    elif noise_type == 'gauss':
        Ns = np.random.normal(scale=std, size=(n, d))
        Nf = np.random.normal(scale=std, size=(n, d))
    else: # considering gumbel case
        noise_scale = np.sqrt(6) * std / np.pi
        Ns = np.random.gumbel(scale=noise_scale, size=(n, d))
        Nf = np.random.gumbel(scale=noise_scale, size=(n, d))

    if noise_effect == 'root_causes':
        Ns = np.zeros((n, d))
    elif noise_effect == 'signal':
        Nf = np.zeros((n, d))

    # computation according to definition of transitive closure (either Floyd-Warshall or zero)
    # refl_trans_clos = np.linalg.inv(I - W)
    A = csc_matrix(I - W.T)
    B = (C + Nf).T
    X = spsolve(A, B) # (X = XW + C + Nf)
    X = X.T + Ns

    # Condition number of (I + transclos(W)) matrix
    cond_num = 0 # np.linalg.cond(refl_trans_clos)

    return X, C, cond_num