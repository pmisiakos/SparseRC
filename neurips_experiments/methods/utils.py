import time 
import cdt 
import pandas as pd
import networkx as nx
import numpy as np

# sparserc
from sparserc.sparserc import sparserc_solver

# DAGMA
from neurips_experiments.methods.dagma import dagma_linear

# notears
import neurips_experiments.methods.notears.notears.utils as notears_utils 
from neurips_experiments.methods.notears.notears.linear import notears_linear

# fGES 
from pycausal.pycausal import pycausal as pc # pycausal package is used in fast greedy equivalence search algorithm (fGES or FGS)

# NoCurl
import neurips_experiments.methods.NoCurl.BPR 
import neurips_experiments.methods.NoCurl.main_efficient 

# golem
from neurips_experiments.methods.golem.src.golem import golem
from neurips_experiments.methods.golem.src.utils.train import postprocess
from neurips_experiments.methods.golem.src.utils.config import get_args as golem_get_args

# varsortability
from neurips_experiments.methods.Varsortability.src.sortnregress import sortnregress
from neurips_experiments.methods.Varsortability.src.varsortability import varsortability

# LiNGAM 
from neurips_experiments.methods.lingam import lingam 



def execute_method(X, method, f, args, pc=None, dataset="synthetic", XF=None):
    if method == 'sparserc':
        start = time.time()
        if(dataset == "synthetic"):
            W_est = sparserc_solver(X, lambda1=0, lambda2=1, epochs=args.sparserc_epochs, omega=args.omega)
            # W_est = sparserc_solver(X, lambda1=0.001, lambda2=10, args=args, epochs=args.sparserc_epochs, omega=args.omega) for fixsup
        elif(dataset == "sachs"):
            W_est = sparserc_solver(X, lambda1=0, lambda2=20, epochs=10000, omega=0.3)
        print(" Time for sparserc was {:.3f}".format(time.time() - start))
        T = time.time() - start
        B_est = W_est != 0

    elif method == 'dagma':
        model = dagma_linear.DAGMA_linear(loss_type='l2')
        start = time.time()
        W_est = model.fit(X, lambda1=0.02, w_threshold=args.omega)
        T = time.time() - start
        B_est = W_est != 0

    elif method == 'notears': 
        start = time.time()
        if(dataset == "synthetic"):
            W_est = notears_linear(X, lambda1=0.001, loss_type='l2', w_threshold=args.omega)
        elif(dataset == "sachs"):
            W_est = notears_linear(X, lambda1=0.00, loss_type='l2', w_threshold=0.3)
        print(" Time for notears was {:.3f}".format(time.time() - start))
        T = time.time() - start
        B_est = W_est != 0

    elif method == 'golem':
        tmp_args = golem_get_args()
        start = time.time()
        W_est = golem(X, 2e-2, 5.0, tmp_args.equal_variances, tmp_args.num_iter,
                        tmp_args.learning_rate, tmp_args.seed, tmp_args.checkpoint_iter) 
        W_est = postprocess(W_est, args.omega) 
        print(" Time for golem was {:.3f}".format(time.time() - start))
        T = time.time() - start
        B_est = W_est != 0

    elif method == 'golem-ev': #execution for sachs dataset
        tmp_args = golem_get_args()
        start = time.time()
        W_est = golem(X, 1e-3, 10, True, tmp_args.num_iter,
                        tmp_args.learning_rate, tmp_args.seed, tmp_args.checkpoint_iter)
        W_est = postprocess(W_est, tmp_args.graph_thres)
        T = time.time() - start
        B_est = W_est != 0

    elif method == 'golem-nv':
        tmp_args = golem_get_args()
        #initializing golem-nv with golem-ev
        start = time.time()
        W_est = golem(X, 1e-3, 10, True, 20000,
                        tmp_args.learning_rate, tmp_args.seed, tmp_args.checkpoint_iter)
        W_est = postprocess(W_est, tmp_args.graph_thres)
        B_init = W_est != 0
        W_est = golem(X, 0, 20, False, 20000,
                        tmp_args.learning_rate, tmp_args.seed, tmp_args.checkpoint_iter, B_init=B_init)
        W_est = postprocess(W_est, tmp_args.graph_thres)
        T = time.time() - start
        B_est = W_est != 0

    elif method == 'sortnregress':
        start = time.time()
        W_est = sortnregress(X)
        T = time.time() - start
        W_est = np.where(np.abs(W_est) > args.omega, W_est, 0) #thresholding
        B_est = W_est != 0

    elif method == 'lingam':
        model = lingam.ICALiNGAM()
        start = time.time()
        model.fit(X)
        T = time.time() - start
        W_est = model.adjacency_matrix_.T
        W_est = np.where(np.abs(W_est) > args.omega, W_est, 0) #thresholding
        B_est = W_est != 0
    
    elif method == 'direct_lingam':
        model = lingam.DirectLiNGAM()
        start = time.time()
        model.fit(X)
        T = time.time() - start
        W_est = model.adjacency_matrix_.T
        W_est = np.where(np.abs(W_est) > args.omega, W_est, 0) #thresholding
        B_est = W_est != 0
    
    elif method == 'pc':
        model = cdt.causality.graph.PC(CItest='gaussian', method_indep='corr')  # Peter-Clark algorithm
        data_frame = pd.DataFrame(X)
        start = time.time()
        output_graph_nc = model.predict(data_frame)
        T = time.time() - start
        W_est = nx.adjacency_matrix(output_graph_nc).todense()
        W_est = np.asarray(W_est).astype(np.float64)
        W_est = np.where(np.abs(W_est) > args.omega, W_est, 0) #thresholding
        B_est = W_est != 0

    else:
        tmp_args = neurips_experiments.methods.NoCurl.main_efficient.get_args()
        tmp_args.graph_threshold = args.omega
        bpr = neurips_experiments.methods.NoCurl.BPR.BPR(tmp_args, pc)
        start = time.time()
        try:
            W_est, h, alpha, rho = bpr.fit(X, method)
        except:
            f.write("Error in method {}\n".format(method))
            # continue
        T = time.time() - start
        print("Done {}".format(method))
        B_est = W_est != 0 if method == 'nocurl' else W_est

    return B_est, W_est, T
    