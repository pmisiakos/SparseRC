from . import data_generation
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler


def get_data(args, n, d, dataset="synthetic"):
    if dataset == "synthetic":
        (a, b) = tuple(args.weight_bounds)
        k = args.edges

        B_true = data_generation.simulate_dag(d, k * d, args.graph_type) # Erdös-Renyi graph simulation with avg degree = k
        W_true = data_generation.simulate_parameter(B_true, w_ranges=((-b, -a), (a, b))) # sampling uniformly the weights

        # data initialization root_causes SEM: X = C(I + \bar(W))
        X, C_true, cond_num = data_generation.sparse_rc_sem(W_true, n, sparsity=args.sparsity, std=args.noise_std, 
                                                    noise_type=args.noise, noise_effect=args.noise_effect, fix_sup=args.fixSup)

        return X, C_true, cond_num, B_true, W_true

    elif (dataset == "sachs"):
        X = np.load('neurips_experiments/data/sachs/data1.npy')
        B_true = np.load('neurips_experiments/data/sachs/DAG1.npy')
        print(B_true)

        print(X.shape)

        return X, 0, 0, B_true, 0

def data_transform(X, args):
    # applying transformation to data (or not)
    if (args.transformation == 'norm'):
        X = normalize(X)
    elif (args.transformation == 'stand'):
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)
    return X
