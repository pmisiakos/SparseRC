import numpy as np
from neurips_experiments import utils
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

def rc_approximation(method, X, W_est, C_true, epsilon=0.1):
    '''
    naive implementation of approximating C in the cases:
    X = (C + N)(I + \bar(W))
    and 
    C(I + \bar(W)) + N

    returns nan or rc nmse
    '''
    if method not in ['sparserc', 'notears', 'golem']:
        return float("nan"), float("nan"), float("nan") # only top-performing algorithms
    else:
        d = W_est.shape[0]
        inverse_refl_trans_clos = np.eye(d) - W_est
        C_est = X @ inverse_refl_trans_clos
        rc_nmse = np.linalg.norm(C_est - C_true) / np.linalg.norm(C_true)

        non_zero_est = np.where(C_est > epsilon * np.max(C_est), 1, 0)
        non_zero_true = np.where(C_true > 0, 1, 0)
        rc_support_tpr = np.sum(non_zero_est * non_zero_true) / np.sum(non_zero_true)

        zero_true = np.where(C_true > 0, 0, 1)
        rc_support_fpr = np.sum(non_zero_est * zero_true) / np.sum(zero_true)

        if not non_zero_true is 0 and not C_true is 0:
            visualize_rc(non_zero_true[:25], non_zero_est[:25], method)

        tp = np.sum(non_zero_est * non_zero_true)
        p = np.sum(non_zero_true)
        fp = np.sum(non_zero_est * zero_true)
        n = np.sum(zero_true)
        print("TP = {}, P = {}, FP = {}, N = {}".format(tp, p, fp, n))

    return rc_nmse, rc_support_tpr, rc_support_fpr


def visualize_rc(non_zero_true, non_zero_est, method):
    gray = cm.get_cmap('gray', 4)
    newcolors = gray(np.linspace(0, 1, 4))
    white = np.array([1, 1, 1, 1])
    black = np.array([0, 0, 0, 1])
    red = np.array([1, 0, 0, 1])
    grey = np.array([0.5, 0.5, 0.5, 1])
    newcolors[0, :] = white
    newcolors[1, :] = grey
    newcolors[2, :] = red
    newcolors[3, :] = black
    custom_cmp = ListedColormap(newcolors)

    l2 = np.where(non_zero_est != 0, 1, 0)

    common_l2 = non_zero_true * l2
    wrong_l2 = l2 - common_l2
    missed_l2 = non_zero_true - common_l2
    l2 = common_l2 + 0.66 * wrong_l2 + 0.33 * missed_l2

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(14, 6)

    ax1.imshow(non_zero_true, cmap=custom_cmp)
    ax1.set_title('Ground Truth')

    ax2.imshow(l2, cmap=custom_cmp)
    ax2.set_title('Estimated')

    fig.suptitle('Root causes')

    plt.savefig('neurips_experiments/plots/root_causes_{}.pdf'.format(method), dpi=1000)


def count_accuracy(B_true, B_est):
    "Copied from NOTEARS repo"
    
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not utils.is_dag(B_est):
            # raise ValueError('B_est should be a DAG')
            print('Warning: B_est is not a DAG') # in order also to evaluate algorithms that do not return a DAG
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}