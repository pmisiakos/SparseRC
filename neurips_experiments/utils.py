import numpy as np
import igraph as ig
import argparse
import random

from threading import Thread
import functools

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', default='gauss', type=str, help='Choices none, gauss, gumbel, uniform')
    parser.add_argument('--noise_std', default=0.01, type=float, help='Noise magnitude')
    parser.add_argument('--noise_effect', default='both', type=str, help='Where the noise is applied. Choices: signal/root_causes')
    parser.add_argument('--sparsity', default=0.1, type=float, help='Probability of data being nonzero at vertex v')
    parser.add_argument('--omega', default=0.09, type=float, help='Thresholding the output matrix of sparserc')

    parser.add_argument('--weight_bounds', default=[0.1, 0.9], nargs='+', type=float, help='initialization of weighted adjacency matrix')
    parser.add_argument('--edges', default=4, type=int, help='graph has k * d edges')

    parser.add_argument('--samples', default=[1000], nargs='+', type=int, help='number of samples')
    parser.add_argument('--nodes', default=[20, 40, 60, 80, 100], nargs='+', type=int, help='number of graph vertices to consider')# [5, 10, 15, 20, 25]
    parser.add_argument('--graph_type', default='ER', type=str, help='Choices ER (Erdös-Renyi), SF (Scale Free)')
    parser.add_argument('--fixSup', default='False', type=str, help='Whether to fix the support of the spectrum')
    parser.add_argument('--methods', default=['sparserc', 'golem', 'notears', 'dagma', 'direct_lingam', 'GES', 'pc', 'lingam', 'CAM', 'nocurl', 'FGS', 'sortnregress', 'MMPC'], nargs='+', type=str, help='methods to compare') 
    parser.add_argument('--transformation', default='None', type=str, help='Whether to normalize/standardize the given signals')
    parser.add_argument('--runs', default=5, type=int, help="how many times to generate the random DAG and run the methods")

    parser.add_argument('--timeout', default=1000, type=int, help='Total allowed runtime for a method')
    parser.add_argument('--table', default='TPR', type=str, help='Choices TPR, SHD')
    parser.add_argument('--sparserc_epochs', default=5000, type=int, help="Number of training epochs of MÖBIUS model")
    parser.add_argument('--legend', default='False', type=str, help='Whether to plot the legend only')
    args = parser.parse_args()

    return parser, args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def timeout(timeout):
    '''
    Timeout function utility
    from: https://stackoverflow.com/questions/21827874/timeout-a-function-windows

    Use: MyResult = timeout(timeout=16)(MyModule.MyFunc)(MyArgs)
    '''
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print ('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def get_filename(parser, args):
    # naming the output files according to the experimental settings 
    dic = vars(args)
    filename = ''
    label = ''
    for key in dic.keys():
        if(key not in ['methods', 'nodes', 'variables', 'legend'] and dic[key]!= parser.get_default(key)):
            filename += '{}_{}_'.format(key, dic[key])

        label += '{} = {}, '.format(key, dic[key])
    filename = filename if len(filename) > 0 else 'default'
    return filename, label
