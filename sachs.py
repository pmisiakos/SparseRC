import neurips_experiments.utils
import neurips_experiments.data.utils
import neurips_experiments.methods.utils
import neurips_experiments.evaluation.utils
import neurips_experiments.plot_experiment

from sparserc.sparserc import sparserc_solver_weight_finder

import time
import numpy as np
import os 

import cdt 
cdt.SETTINGS.rpath = 'C:/Program Files/R/R-4.2.1/bin/Rscript'

from pycausal.pycausal import pycausal as pc

def run_real(pc=None):    
    
    if not os.path.exists("results/{}/".format(filename)):
        os.makedirs("results/{}/".format(filename))

    with open('results/{}.csv'.format(filename), 'a') as f:

        current = {}
        avgT = {}

        for key in args.methods:
            current[key] = []
            avgT[key] = []

        # graph initialization
        start = time.time()
        X, _, _, B_true, _ = neurips_experiments.data.utils.get_data(args, 0, 0, dataset="sachs")
        print("\n\nData generation process done. Time: {:.3f}\n\n".format(time.time() - start))

        # computing root cause sparsity of sachs dataset
        n, d = X.shape
        W_est = sparserc_solver_weight_finder(X, B_true)
        C = X @ (np.eye(d) - W_est)
        c = np.abs(C).max()
        m = np.mean(np.abs(C))
        C_bin = np.where(np.abs(C) > m, 1, 0)
        N = np.count_nonzero(C_bin)
        print("Number of non-zeros is {} / {}".format(N, n*d))
        print("max value is {:.3f}, mean value {:.3f}".format(c, m))
        neurips_experiments.plot_experiment.histogram(np.abs(C))
        
        # causal discovery algorithms
        for method in args.methods:
            B_est, W_est, T = neurips_experiments.methods.utils.execute_method(X, method, f, args, pc, dataset="sachs")
            
            neurips_experiments.evaluation.utils.compute_metrics(method, current, filename, 0, T, X, 0, B_true, 0, B_est, W_est, args)

            # Create two subplots 
            neurips_experiments.plot_experiment.visualize(B_true, B_est, method=method, args=args)

        # save average results in csv
        neurips_experiments.evaluation.utils.save_results(current, f, args)

if __name__ == '__main__':
    parser, args = neurips_experiments.utils.get_args()
    print(vars(args))

    filename = "sachs"
    pc = pc()
    pc.start_vm()
    run_real(pc)
    pc.stop_vm()