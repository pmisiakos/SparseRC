import argparse
import sys

import yaml


def load_yaml_config(path):
    """Load the config file in yaml format.

    Args:
        path (str): Path to load the config file.

    Returns:
        dict: config.
    """
    with open(path, 'r') as infile:
        return yaml.safe_load(infile)


def save_yaml_config(config, path):
    """Load the config file in yaml format.

    Args:
        config (dict object): Config.
        path (str): Path to save the config.
    """
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def get_args():
    """Add arguments for parser.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    # parser = argparse.ArgumentParser()

    class Object(object):
        pass

    args = Object()

    add_dataset_args(args)
    add_model_args(args)
    add_training_args(args)
    add_other_args(args)

    # return parser.parse_args(args=sys.argv[1:])
    return args


def add_dataset_args(args):
    """Add dataset arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """

    args.n=1000
    args.d = 20
    args.graph_type = 'ER'
    args.degree = 4
    args.noise_type = 'gaussian_ev'
    args.B_scale = 1.0

    # parser.add_argument('--n',
    #                 type=int,
    #                 default=1000,
    #                 help="Number of samples.")

    # parser.add_argument('--d',
    #                     type=int,
    #                     default=20,
    #                     help="Number of nodes.")

    # parser.add_argument('--graph_type',
    #                     type=str,
    #                     default='ER',
    #                     help="Type of graph ('ER' or 'SF').")

    # parser.add_argument('--degree',
    #                     type=int,
    #                     default=4,
    #                     help="Degree of graph.")

    # parser.add_argument('--noise_type',
    #                     type=str,
    #                     default='gaussian_ev',
    #                     help="Type of noise ['gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'].")

    # parser.add_argument('--B_scale',
    #                     type=float,
    #                     default=1.0,
    #                     help="Scaling factor for range of B.")


def add_model_args(args):
    """Add model arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """

    args.init=True
    args.lambda_1 = 0.0
    args.lambda_2 = 0.0
    args.equal_variances = True

    # parser.add_argument('--init',
    #                     dest='init',
    #                     action='store_true',
    #                     help="Whether to initialize the optimization with a given weighted matrix.")

    # parser.add_argument('--init_path',
    #                     type=str,
    #                     default=None,
    #                     help="Path of weighted matrix for initialization. Set to None to disable.")

    # parser.add_argument('--lambda_1',
    #                     type=float,
    #                     default=0.0,
    #                     help="Coefficient of L1 penalty.")

    # parser.add_argument('--lambda_2',
    #                     type=float,
    #                     default=0.0,
    #                     help="Coefficient of DAG penalty.")

    # parser.add_argument('--equal_variances',
    #                     dest='equal_variances',
    #                     action='store_true',
    #                     help="Assume equal noise variances for likelibood objective.")

    # parser.add_argument('--non_equal_variances',
    #                     dest='equal_variances',
    #                     action='store_false',
    #                     help="Assume non-equal noise variances for likelibood objective.")


def add_training_args(args):
    """Add training arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """
    args.learning_rate = 1e-3
    args.num_iter = 1e+5
    args.checkpoint_iter = 5000

    # parser.add_argument('--learning_rate',
    #                     type=float,
    #                     default=1e-3,
    #                     help="Learning rate of Adam optimizer.")

    # parser.add_argument('--num_iter',
    #                     type=int,
    #                     default=1e+5,
    #                     help="Number of iterations for training.")

    # parser.add_argument('--checkpoint_iter',
    #                     type=int,
    #                     default=5000,
    #                     help="Number of iterations between each checkpoint. Set to None to disable.")


def add_other_args(args):
    """Add other arguments for parser.

    Args:
        parser (argparse.ArgumentParser): Parser.
    """
    args.seed = 1
    args.graph_thres = 0.3

    # parser.add_argument('--seed',
    #                     type=int,
    #                     default=1,
    #                     help="Random seed.")

    # parser.add_argument('--graph_thres',
    #                     type=float,
    #                     default=0.3,
    #                     help="Threshold for weighted matrix.")
