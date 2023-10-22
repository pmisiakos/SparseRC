#!/bin/bash
# # All experiments

# # Nodes (Default)
python experiment.py --nodes 20 40 60 80 100

# Sensitivity analysis of hyperparameters
python experiment.py --graph_type SF --nodes 100 
python experiment.py --edges 10 --nodes 100 
python experiment.py --sparsity 0.5 --nodes 100 
python experiment.py --weight_bounds 0.5 2 --nodes 100 --omega 0.4 
python experiment.py --noise_std 0.1 --nodes 100 
python experiment.py --noise gumbel --nodes 100 
python experiment.py --transformation stand --nodes 100 
python experiment.py --fixSup True --nodes 100 

# # # Samples
python experiment.py --samples 100 200 500 800 --nodes 100 --methods sparserc dagma golem notears nocurl GES MMPC pc #sortnregress, lingam, direct_lingam, fGES ValueError + CAM timeout

# analysis of edge density
for d in 5 10 15 20; do
    echo python experiment.py --edges $d --nodes 100 --sparsity 0.1 --sparserc_epochs 10000  --methods sparserc notears golem
    python experiment.py --edges $d --nodes 100 --sparsity 0.1 --sparserc_epochs 10000 --methods sparserc notears golem
done

# compare with LiNGAM and DirectLiNGAM
python experiment.py --noise_effect root_causes --nodes 20 40 60 80 100  --methods sparserc lingam direct_lingam

# Large number of nodes
python experiment.py  --sparsity 0.05  --nodes 200  --samples 500  --methods sparserc notears golem --runs 1 --timeout 50000
python experiment.py  --sparsity 0.05  --nodes 500  --samples 1000 --methods sparserc notears golem --runs 1 --timeout 50000
python experiment.py  --sparsity 0.05  --nodes 1000 --samples 5000 --methods sparserc notears golem --runs 1 --timeout 50000
python experiment.py  --sparsity 0.05  --nodes 2000  --samples 10000 --methods sparserc notears --runs 1 --timeout 50000    # golem time-out 
python experiment.py  --sparsity 0.05  --nodes 3000  --samples 10000 --methods sparserc --runs 1 --timeout 50000            # notears, golem time-out

# real dataset
python sachs.py 