
import sys
import os

# appending neurips_experiments to PATH so we can directly execute plot_experiment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from neurips_experiments import utils
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

# adding gillsans font
from matplotlib import font_manager
font_dirs = ['neurips_experiments/plots/fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

label_methods = {
        'sparserc': 'SparseRC', 
        'golem' : 'GOLEM',
        'dagma' : 'DAGMA',
        'notears': 'NOTEARS', 
        'nocurl' : 'DAG-NoCurl', 
        'lingam' : 'LiNGAM',
        'direct_lingam' : 'DirectLiNGAM',
        'GES': 'GES', 
        'MMPC': 'MMHC', 
        'CAM' : 'CAM', 
        'FGS' : 'fGES',
        'sortnregress': 'sortnregress',
        'pc' : 'PC'
    }

def histogram(C):
    with plt.style.context('ggplot'):
        plt.rcParams['font.family'] = 'gillsans'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'

        plt.figure()
        plt.hist(C.flatten(), 100, color='blue')
        plt.xlabel('$|c_{ij}|$', fontsize=28, color='black')
        plt.ylabel('Count of values', fontsize=28, color='black')
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.grid(axis='y', color='white')
        plt.grid(axis='x', color='#e5e5e5')
        # plt.legend(frameon=False, fontsize=18) 
        plt.tight_layout()
        plt.savefig('neurips_experiments/plots/plot_{}.pdf'.format('sachs_root_values_histogram'))

        plt.figure()
        plt.imshow(C.T, extent = [0, 852, 0, 10], aspect = 852/40, cmap='Blues')
        plt.grid(axis='y', color='#e5e5e5')
        plt.xlabel('Rows (samples)', color='black')
        plt.ylabel('Columns (nodes)', color='black')
        plt.tick_params( bottom=False, labelleft=False, labelbottom=False)
        plt.yticks(range(11))
        # plt.legend(frameon=False, fontsize=18) 
        # plt.tight_layout()
        plt.savefig('neurips_experiments/plots/plot_{}.pdf'.format('sachs_root_spikes'), bbox_inches='tight')


def visualize(ground_truth, approximated, method='sparserc', filename='', args=None):
    d = ground_truth.shape[0]
    
    with plt.style.context('ggplot'):
        plt.rcParams['font.family'] = 'gillsans'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'
        
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

        approximated = np.where(approximated != 0, 1, 0)

        common_approximated = ground_truth * approximated
        wrong_approximated = approximated - common_approximated
        missed_approximated = ground_truth - common_approximated 
        approximated = common_approximated + 0.66 * wrong_approximated + 0.33 * missed_approximated

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(14, 4)

        ax1.imshow(ground_truth, cmap=custom_cmp)
        ax1.grid(False)
        ax1.add_patch(Rectangle((-0.5,-0.5), d - 0.15, d - 0.15, linewidth=1, edgecolor='black', facecolor='none'))
        ax1.axis('off')
        ax1.set_title('Ground Truth')

        ax2.imshow(approximated, cmap=custom_cmp)
        ax2.grid(False)
        ax2.add_patch(Rectangle((-0.5,-0.5), d - 0.15, d - 0.15, linewidth=1, edgecolor='black', facecolor='none'))
        ax2.axis('off')
        ax2.set_title(label_methods[method])

        if args is None:
            plt.savefig('neurips_experiments/plots/matrix_comparison/matrix_comparison_sachs.png')
        else:
            plt.savefig('neurips_experiments/plots/matrix_comparison/matrix_comparison_{}_{}.png'.format(filename, method), dpi=1000)


def plot_accuracy(avg, std, x_axis, methods, param='nodes', filename='default', legend=False):
    full = 'MMPC' in methods # full version of plot with all methods

    linewidth = {}
    for method in color_methods.keys():
        linewidth[method] = 1.5
    linewidth['sparserc'] = 3

    with plt.style.context('ggplot'):
        plt.rcParams['font.family'] = 'gillsans'
        plt.rcParams['xtick.color'] = 'black'
        plt.rcParams['ytick.color'] = 'black'

        for i, label in enumerate(['SHD', 'TPR', 'NNZ', 'SID', 'NMSE', 'Time (s)', '$\mathbf{C}$ TPR', '$\mathbf{C}$ NMSE', '$\mathbf{C}$ FPR', 'FPR']):
            if param != 'weight_bounds':
                if (not legend):
                    plt.figure()
                    for method in methods:
                        if(len(avg[method] > 0)):
                            plt.plot(x_axis, avg[method][:, i], label = label_methods[method], color=color_methods[method], linewidth=linewidth[method])
                            plt.fill_between(x_axis, avg[method][:, i] - std[method][:, i], avg[method][:, i] + std[method][:, i], color=color_methods[method], alpha=.1)

                    # if filename == 'default':
                    plt.ylabel(label, fontsize=30, color='black')
                    # plt.xlabel('Number of sample sizes' if args.value=='samples' else 'Number of vertices', fontsize=20)
                    # if (label == 'Time (s)'): #if (label == 'SID' and not full) or (full and label == 'NMSE'):
                    plt.xlabel('Number of nodes' if param=='nodes' else 'Number of samples' if param=='samples' else 'Sparsity %' if param=='sparsity' else 'Noise std.' if param=='noise_std' else 'Avg. degree', fontsize=28, color='black')
                    plt.xticks(x_axis, fontsize=22)
                    plt.yticks(fontsize=22)
                    plt.grid(axis='y', color='white')
                    plt.grid(axis='x', color='#e5e5e5')
                    # plt.legend(frameon=False, fontsize=18) 
                    # if label == 'SHD' and param != 'nodes':
                    #     plt.ylim([0, 2 * 100])
                    if label in lims.keys():
                        plt.ylim(lims[label])
                    plt.tight_layout()

                    fullname = '_full' if full else ''
                    plt.savefig('neurips_experiments/plots/plot{}_{}_{}.pdf'.format(fullname, filename, file_label[label]))

                # # only print legend
                elif (i == 0):
                    plt.figure()

                    plt.rcParams['axes.facecolor']='white'
                    plt.rcParams['savefig.facecolor']='white'
                    for method in methods:
                        if(len(avg[method] > 0)):
                            plt.plot([], [], label = label_methods[method], color=color_methods[method], linewidth=linewidth[method])
                    plt.xticks([])
                    plt.yticks([])
                    plt.legend(frameon=False, fontsize=20)
                    plt.tight_layout()
                    fullname = '_full' if full else ''
                    plt.savefig('neurips_experiments/plots/plot{}_{}_legend_only.pdf'.format(fullname, filename), bbox_inches='tight')
            
            else: 
                bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]    
                y, x = np.meshgrid(bounds, bounds)
                
                z = {}
                for method in methods:
                    z[method] = np.zeros(x.shape)
                
                for j in range(len(x_axis)):
                    v = x_axis[j]
                    v0 = int(10 * v[0]) - 1
                    v1 = int(10 * v[1]) - 1
                    for method in methods:
                        z[method][v0, v1] = avg[method][j, i]
                    
                for method in methods:
                    if(len(avg[method] > 0)):
                        plt.figure()
                        z_min, z_max = np.abs(z[method]).min(), np.abs(z[method]).max()
                        plt.pcolormesh(x, y, z[method], cmap='RdBu', vmin=z_min, vmax=z_max)# avg[method][:, i], label = label_methods[method], color=color_methods[method], linewidth=linewidth[method])
                        plt.colorbar()
                        plt.xlabel('Lower bound a')
                        plt.ylabel('Upper bound b')
                        for i in range(z[method].shape[0]):
                            for j in range(z[method].shape[1]):
                                plt.text(j, i, '{:.2f}'.format(z[method][i, j]), ha='center', va='center')
                        plt.savefig('neurips_experiments/plots/plot_{}_{}_{}.png'.format(filename, method, label))
                



def plot_accuracy_vs_param(args, param='sparsity'):
    dic = vars(args)

    avg = {}
    std = {}
    for key in methods:
        avg[key] = []
        std[key] = []

    if param == 'nodes':
        values = args.nodes
    elif param == 'samples':
        values = args.samples
    elif param == 'sparsity':
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    elif param == 'weight_bounds':
        bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]    
        values = [[a, b] for a in bounds for b in bounds if a <= b]
    elif param == 'noise_std':
        values = [0.01, 0.1, 0.2, 0.5, 1, 2, 5]
    elif param == 'edges':
        values = [5, 10, 15, 20]
    else:
        print("case not covered")

    if param == 'nodes' or param == 'samples':
        # finding the name of the output files according to the experimental settings 
        filename, _ = utils.get_filename(parser, args)

        with open('results/{}.csv'.format(filename), 'r') as f:
            for line in f:
                info = line.split(',')
                for method in methods:
                    if(info[0] == 'Acc {} is'.format(method)):
                        avg[method].append([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[7]), float(info[6]), float(info[8]), float(info[9]), float(info[10])])
                    elif(info[0] == 'Std {} is'.format(method)):
                        std[method].append([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[7]), float(info[6]), float(info[8]), float(info[9]), float(info[10])])
    else:
        for v in values:
            if v == parser.get_default(param):
                filename = 'default'
            else:
                if param in ['weight_bounds', 'edges']:
                    filename = '{}_{}_'.format(param, v)
                else:
                    filename = '{}_{:.1f}_'.format(param, v)

            with open('results/{}.csv'.format(filename), 'r') as f:
                for line in f:
                    info = line.split(',')
                    for method in methods:
                        if(info[0] == 'Acc {} is'.format(method)):
                            avg[method].append([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[7]), float(info[6]), float(info[8]), float(info[9]), float(info[10])])
                        elif(info[0] == 'Std {} is'.format(method)):
                            std[method].append([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[7]), float(info[6]), float(info[8]), float(info[9]), float(info[10])])

        filename = param

    for method in methods:
        avg[method] = np.array(avg[method])
        std[method] = np.array(std[method])

    x_axis = values
    plot_accuracy(avg, std, x_axis, methods, param=param, filename=filename, legend=(args.legend == 'True'))


if __name__ == '__main__':
    parser, args = utils.get_args()

    samples = args.samples #[200, 400, 600, 800, 1000] if(args.value == 'samples') else [400]
    variables =  args.nodes # if(args.value == 'variables') else [10] 
    noise = args.noise
    runs = args.runs 
    (a, b) = tuple(args.weight_bounds)
    k = args.edges
    methods = args.methods

    color_methods = {
        'sparserc': 'black', 
        'golem' : 'cyan',
        'dagma' : 'C7',
        'notears' : 'C2', 
        'nocurl' : 'C5', 
        'lingam' : 'navy',
        'direct_lingam' : 'green',
        'GES': 'C4', 
        'MMPC': 'sienna', 
        'CAM' : 'purple', 
        'FGS' : 'teal',
        'sortnregress': 'pink',
        'pc' : 'dodgerblue'
    }


    file_label = {
        'SHD' : 'shd',
        'TPR' : 'tpr',
        'NNZ' : 'nnz', 
        'SID' : 'sid', 
        'NMSE': 'nmse', 
        '$\mathbf{C}$ TPR' : 'c_tpr', 
        'Time (s)' : 'time', 
        '$\mathbf{C}$ NMSE' : 'c_nmse',
        '$\mathbf{C}$ FPR' : 'c_fpr',
        'FPR' : 'fpr'
    }

    lims = { 
        'Time (s)' :  [0, 500],
        '$\mathbf{C}$ TPR' : [0.75, 1]
    }

    if len(args.nodes) > 1:
        plot_accuracy_vs_param(args, param='nodes')
    else:
        lims['SHD'] = [0, 500]
        plot_accuracy_vs_param(args, param='samples')
    
    # plot_accuracy_vs_param(args, param='sparsity')

    # plot_accuracy_vs_param(args, param='noise_std')

    # plot_accuracy_vs_param(args, param='weight_bounds')

    # plot_accuracy_vs_param(args, param='edges')

