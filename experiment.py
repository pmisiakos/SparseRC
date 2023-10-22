import neurips_experiments.utils
import neurips_experiments.data.utils
import neurips_experiments.methods.utils
import neurips_experiments.evaluation.utils
import neurips_experiments.plot_experiment

from tqdm import tqdm
import time
import pandas as pd
import os
import numpy as np

# varsortability
from neurips_experiments.methods.Varsortability.src.varsortability import varsortability

# Causal Discovery toolbox + setup R path
import cdt 
cdt.SETTINGS.rpath = 'C:/Program Files/R/R-4.2.1/bin/Rscript'

# fGES 
from pycausal.pycausal import pycausal as pc # pycausal package is used in fast greedy equivalence search algorithm (fGES or FGS)
pc = pc()
pc.start_vm()


if __name__ == '__main__':
    parser, args = neurips_experiments.utils.get_args()
    print(vars(args))

    # naming the output files according to the experimental settings
    filename, label = neurips_experiments.utils.get_filename(parser, args)

    # make directory to put results
    if not os.path.exists("results/{}/".format(filename)):
        os.makedirs("results/{}/".format(filename))

    for n in args.samples:
        for d in args.nodes:
            with open('results/{}.csv'.format(filename), 'a') as f:
                f.write('{}\n'.format(label))

                print('samples = {}, nodes = {}, noise = {}'.format(n, d, args.noise))
                f.write('samples = {}, nodes = {}, noise = {}\n'.format(n, d, args.noise))

                current = {}
                avgT = {}

                for key in args.methods:
                    current[key] = []
                    avgT[key] = []

                avg_cond_num = 0 
                avg_varsortability = 0

                for r in tqdm(range(args.runs)):

                    # graph initialization
                    start = time.time()
                    X, C_true, cond_num, B_true, W_true = neurips_experiments.data.utils.get_data(args, n, d)
                    
                    df = pd.DataFrame(C_true)
                    df.to_csv('results/{}/C_run_{}_{}.csv'.format(filename, r, d), header=None, index=False)
                    df = pd.DataFrame(X)
                    df.to_csv('results/{}/X_run_{}_{}.csv'.format(filename, r, d), header=None, index=False)
                    df = pd.DataFrame(W_true)
                    df.to_csv('results/{}/W_true_run_{}_{}.csv'.format(filename, r, d), header=None, index=False)

                    print("\n\nData generation process done. Time: {:.3f}\n\n".format(time.time() - start))

                    # computing average condition number of (I + \bar(W))
                    avg_cond_num += cond_num

                    # normalizes or standardizes data if supposed to
                    X = neurips_experiments.data.utils.data_transform(X, args) 

                    # computing varsortability of dataset '
                    print(X.shape, W_true.shape)
                    if d < 200:
                        avg_varsortability += varsortability(X, W_true)

                    # causal discovery algorithms
                    for method in args.methods:
                        # try: 
                        #     B_est, W_est, T = neurips_experiments.utils.timeout(timeout=args.timeout)(neurips_experiments.methods.utils.execute_method)(X, method, f, args, pc)
                        # except: # in case of time-out
                        #     B_est, W_est, T = np.zeros((d,d)), np.zeros((d,d)), args.timeout
                        B_est, W_est, T = neurips_experiments.methods.utils.execute_method(X, method, f, args, pc)
                        print(B_est)
                        # save result for future reference
                        df = pd.DataFrame(W_est)
                        df.to_csv('results/{}/W_est_{}_run_{}_{}.csv'.format(filename, method, r, d), header=None, index=False)
                        
                        # Create two subplots 
                        neurips_experiments.plot_experiment.visualize(B_true, W_est, method, filename, args)

                        neurips_experiments.evaluation.utils.compute_metrics(method, current, filename, r, T, X, C_true, B_true, W_true, B_est, W_est, args)

                # computing varsortability of dataset 
                neurips_experiments.evaluation.utils.compute_varsortability(avg_varsortability, f, args)

                # outputting cond num of matrices (I + transclos(W)) for all runs
                neurips_experiments.evaluation.utils.cond_num(avg_cond_num, f, args)

                # save average results in csv
                neurips_experiments.evaluation.utils.save_results(current, f, args)

pc.stop_vm()
