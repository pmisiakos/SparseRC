import numpy as np
import neurips_experiments.evaluation.evaluation 
import neurips_experiments.utils
import pandas as pd
import cdt 


def compute_metrics(method, current, filename, r, T, X, C_true, B_true, W_true, B_est, W_est, args):
    d = X.shape[1]
    c_nmse, c_tpr, c_fpr = neurips_experiments.evaluation.evaluation.rc_approximation(method, X, W_est, C_true)
    nmse = np.linalg.norm(W_est - W_true) / np.linalg.norm(W_true)
    acc = neurips_experiments.evaluation.evaluation.count_accuracy(B_true, B_est)
    shd = cdt.metrics.SHD(B_true, B_est, double_for_anticausal=False)
    
    try: # sid computation assumes acyclic graph
        if(not  neurips_experiments.utils.is_dag(B_est)):
            print("Warning, output is not a DAG, SID doesn't make sense")
        sid = neurips_experiments.utils.timeout(timeout=100)(cdt.metrics.SID)(B_true, B_est) 
    except:
        sid = float("nan")
    current[method].append([shd, acc['tpr'], acc['nnz'], sid, nmse, c_tpr, T, c_nmse, c_fpr, acc['fpr']])
    print("Results, SHD, TPR, NNZ, SID, NMSE, C_TPR, T, C_NMSE, C_FPR, FPR")
    print("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}"
            .format(method, current[method][r][0], current[method][r][1], current[method][r][2], current[method][r][3], current[method][r][4], current[method][r][5], current[method][r][6], current[method][r][7], current[method][r][8], current[method][r][9]))

    # looking at weights
    if d > 100:
        df = pd.DataFrame(W_est)
        df.to_csv('results/W_est_{}_nodes_{}_{}.csv'.format(filename, d, method), header=None, index=False)
        df = pd.DataFrame(W_true)
        df.to_csv('results/W_true_{}_nodes_{}_{}.csv'.format(filename, d, method), header=None, index=False)


def compute_varsortability(avg_varsortability, f, args):
    # computing varsortability of dataset 
    avg_varsortability = avg_varsortability / args.runs
    print("Avg Varsortability, {:.3f}".format(avg_varsortability))
    f.write("Avg Varsortability, {:.3f}\n".format(avg_varsortability))

def cond_num(avg_cond_num, f, args):
    avg_cond_num = avg_cond_num / args.runs
    print('Avg cond num of (I + transclos(W)) is {:.3f}'.format(avg_cond_num))
    f.write('Avg cond num of (I + transclos(W)) is {:.3f}\n'.format(avg_cond_num))

def save_results(current, f, args):      
    # Log results
    avg = {}
    std = {}

    f.write("Results, SHD, TPR, NNZ, SID, NMSE, C_TPR, T, C_NMSE, C_FPR, FPR\n")

    for method in args.methods:
        avg[method] = np.mean(current[method], axis=0)
        std[method] = np.std(current[method], axis=0)
        
        f.write("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(method, avg[method][0], avg[method][1], avg[method][2], avg[method][3], avg[method][4], avg[method][5], avg[method][6], avg[method][7], avg[method][8], avg[method][9]))
        f.write("Std {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n".format(method, std[method][0], std[method][1], std[method][2], std[method][3], std[method][4], std[method][5], std[method][6], std[method][7], std[method][8], std[method][9]))
        print("Acc {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(method, avg[method][0], avg[method][1], avg[method][2], avg[method][3], avg[method][4], avg[method][5], avg[method][6], avg[method][7], avg[method][8], avg[method][9]))
        print("Std {} is, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(method, std[method][0], std[method][1], std[method][2], std[method][3], std[method][4], std[method][5], std[method][6], std[method][6], std[method][8], std[method][9]))
