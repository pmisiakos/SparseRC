import sys
import os

# appending neurips_experiments to PATH so we can directly execute plot_experiment
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from neurips_experiments import utils

if __name__ == '__main__':
    parser, args = utils.get_args()

    methods = args.methods
    if(args.table == 'TPR'):
        ind = 1 # reporting TPR
    elif(args.table == 'SID'):
        ind = 3 # reporting SID
    elif(args.table == 'SHD'):
        ind = 0 # reporting SHD
    else:
        ind = 5 # reporting runtime

    # finding the name of the output files according to the experimental settings 
    dic = vars(args)
    filename = ''
    table = {
        'default'           :'1.  & \color{NavyBlue}{Default settings}    &                                                                                              &                                                                                       ', 
        'graph_type'        :'2.  & Graph type                            & \color{NavyBlue}{Erd\\"os-Renyi}                                                              &  Scale-free                                                                           ',
        'edges'             :'3.  & Edges / Vertices                      & \color{NavyBlue}{$4$}                                                                        &   $10$                                                                                 ',
        'weight_bounds'     :'4.  & Larger weights in $\mb{A}$            & \color{NavyBlue}{$(0.1,0.9)$}                                                                &   $(0.5, 2)$                                                                          ',
        'sparsity'          :'5.  & Dense root causes $\mb{C}$            & \color{NavyBlue}{$p=0.1$}                                                                    &   $p=0.5$                                                                             ',
        'noise_std'         :'6.  & $\mb{N}_c,\mb{N}_x$ deviation         & \color{NavyBlue}{$\sigma=0.01$}                                                              &  $\sigma=0.1$                                                                         ',
        'noise'             :'7.  & $\mb{N}_c,\mb{N}_x$ distribution      & \color{NavyBlue}{Gaussian}                                                                   &   Gumbel                                                                              ',
        'transformation'    :'8. & Standardization                        & \color{NavyBlue}{No}                                                                         &   Yes                                                                                 ',
        'samples'           :'9. & Samples                                & \color{NavyBlue}{$n=1000$}                                                                   &   $n=100$                                                                             ',
        'fixSup'            :'10. & Fixed support                         & \color{NavyBlue}{No}                                                                         &   Yes                                                                                 '
    }

    for key in dic.keys():
        if(key not in ['methods', 'nodes', 'table'] and dic[key]!= parser.get_default(key)):
            filename += '{}_{}_'.format(key, dic[key])
        if(key not in ['methods', 'nodes', 'table', 'sparserc_epochs', 'omega'] and dic[key]!= parser.get_default(key)):
            table_entry = table[key] # only one setting is altered in each experiment. 
                                     # we correspond it to the initial latex entry. 
    if len(filename) == 0:
        filename = 'default'
        table_entry = table['default']

    avg = {}
    std = {}
    for key in methods:
        avg[key] = []
        std[key] = []
    
    varsortability = []

    with open('results/{}.csv'.format(filename), 'r') as f:
        for line in f:
            info = line.split(',')
            for method in methods:
                if(info[0] == 'Acc {} is'.format(method)):
                    avg[method].append([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[7])])
                elif(info[0] == 'Std {} is'.format(method)):
                    std[method].append([float(info[1]), float(info[2]), float(info[3]), float(info[4]), float(info[5]), float(info[7])])
                elif(info[0] == 'Avg Varsortability'):
                    varsortability.append(float(info[1]))

    pre_last = len(avg[methods[0]]) - 1 # 1 if len(avg[methods[0]]) > 1 else 0 
    best_tpr = max([avg[m][pre_last][1] for m in methods])
    best_shd = min([avg[m][pre_last][0] for m in methods])
    best_sid = min([avg[m][pre_last][3] for m in methods])
    best_time = min([avg[m][pre_last][5] for m in methods])
        
    # adding varsortability measurement
    # result = '{}& $ {:.2f} $'.format(table_entry, varsortability[pre_last]) 
    result = '{} '.format(table_entry)


    SHD_fail = 400 if args.edges == parser.get_default("edges") else args.edges * args.nodes[-1]
    TPR_fail = 0.5

    for method in methods:
        if(ind == 3 and np.isnan(avg[method][pre_last][ind])):
            result += '&       ' + ' ' + '  SID time-out     '

        elif((avg[method][pre_last][0] > SHD_fail and args.table == 'SHD' and ind == 0 and avg[method][pre_last][1] > 0.7)): 
            result += '&       ' + ' ' + '  failure \color{ForestGreen}{' + '({:.3f})'.format(avg[method][pre_last][1]) + "}"

        elif((avg[method][pre_last][1] < TPR_fail and args.table == 'TPR') or 
           (avg[method][pre_last][0] > SHD_fail and args.table == 'SHD')): 
           result += '&       ' + ' ' + '  failure          '

        elif((avg[method][pre_last][1] == best_tpr and args.table == 'TPR') or 
           (avg[method][pre_last][0] == best_shd and args.table == 'SHD') or 
           (avg[method][pre_last][3] == best_sid and args.table == 'SID') or
           (avg[method][pre_last][5] == best_time and args.table == 'time')): 
           result += '&  $\\bm' + '{' + '{:.2f}\pm{:.2f}'.format(avg[method][pre_last][ind], std[method][pre_last][ind]) + '}$  '

        else:
           result += '&  $    {:.2f}\pm{:.2f} $  '.format(avg[method][pre_last][ind], std[method][pre_last][ind]) 
        
    result += '\\\\ \n'
    with open('results/table.tex', 'a') as f:
        f.write(result)