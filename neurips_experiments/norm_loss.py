import numpy as np
import pandas as pd
import os 
import sys

# appending neurips_experiments to PATH so we can directly execute plot_experiment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from neurips_experiments import utils

if __name__ == '__main__':
    parser, args = utils.get_args()

    print(vars(args))
    samples = args.samples # [200, 400, 600, 800, 1000] #if(args.value == 'samples') else [400]
    variables =  args.nodes #if(args.value == 'variables') else [10] 
    d = variables[0]
    n = samples[0]
    method = args.methods[0]
    noise = args.noise
    runs = args.runs 
    (a, b) = tuple(args.weight_bounds)
    k = args.edges
    methods = args.methods
    T = 5 # number of timesteps in dynamic DAG
    fix_sup = args.fixSup

    # naming the output files according to the experimental settings 
    dic = vars(args)
    filename = ''
    label = ''
    for key in dic.keys():
        if(key not in ['methods','nodes','variables'] and dic[key]!= parser.get_default(key)):
            filename += '{}_{}_'.format(key, dic[key])

        label += '{} = {}, '.format(key, dic[key])
    filename = filename if len(filename) > 0 else 'default'


    # looking at weights
    df = pd.read_csv('results/W_est_{}_nodes_{}_{}.csv'.format(filename, d, method), header=None)
    W_est = df.to_numpy()
    
    df = pd.read_csv('results/W_true_{}_nodes_{}_{}.csv'.format(filename, d, method), header=None)
    W_true = df.to_numpy()

    l11_loss = np.sum(np.abs(W_est - W_true))
    l1_loss = np.max(np.abs(W_est - W_true))
    l2_loss = np.linalg.norm(W_est - W_true)
    nmse = np.linalg.norm(W_est - W_true) / np.linalg.norm(W_true)


    print("L1,1 norm loss is {:.5f}".format(l11_loss))
    print("Edges are {}. Avg L1,1 norm loss is {:.5f}".format(2 * d, l11_loss / (2 * d)))

    print("L1 norm (max-abs) loss is {:.5f}".format(l1_loss))

    print("L2 norm loss is {:.5f}".format(l2_loss))
    print("Edges are {}. Avg L2 norm loss is {:.5f}".format(2 * d, l2_loss / (2 * d)))

    print("NMSE is {:.5f}".format(nmse))
    with open("results/norm_loss.tex", 'a') as f:
        f.write("$d={},\,n={}$ & $ {:.3f} $ &  $ {:.3f} $ & $ {:.3f} $ & $ {:.3f} $\\\\\n".format(d, n, l11_loss / (2 * d), l1_loss, l2_loss / (2 * d), nmse))
    

                    