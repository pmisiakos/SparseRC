# Reporting results in table and plots format. 
# # printing SHD table in latex format
python neurips_experiments/print_table_tex.py --table SHD --nodes 100 --methods sparserc golem notears dagma direct_lingam pc GES
python neurips_experiments/print_table_tex.py --graph_type SF --nodes 100 --table SHD --methods sparserc golem notears dagma direct_lingam pc GES
python neurips_experiments/print_table_tex.py --edges 10 --nodes 100 --table SHD --methods sparserc golem notears dagma direct_lingam pc GES
python neurips_experiments/print_table_tex.py --weight_bounds 0.5 2 --nodes 100 --omega 0.4 --table SHD --methods sparserc golem notears dagma direct_lingam pc GES
python neurips_experiments/print_table_tex.py --sparsity 0.5 --nodes 100 --table SHD --methods sparserc golem notears dagma direct_lingam pc GES
python neurips_experiments/print_table_tex.py --noise_std 0.1 --nodes 100 --table SHD --methods sparserc golem notears dagma direct_lingam pc GES
python neurips_experiments/print_table_tex.py --noise gumbel --nodes 100 --table SHD --methods sparserc golem notears dagma direct_lingam pc GES
python neurips_experiments/print_table_tex.py --transformation stand --nodes 100 --table SHD --methods sparserc golem notears dagma direct_lingam pc GES
python neurips_experiments/print_table_tex.py --samples 100 --nodes 100  --table SHD --methods sparserc golem notears dagma pc GES #sortnregress ValueError
python neurips_experiments/print_table_tex.py --fixSup True --table SHD --methods sparserc golem notears dagma direct_lingam pc GES

# printing Time table in latex format
python neurips_experiments/print_table_tex.py --table time --nodes 100 --methods sparserc golem notears
python neurips_experiments/print_table_tex.py --graph_type SF --nodes 100 --table time --methods sparserc golem notears
python neurips_experiments/print_table_tex.py --edges 10 --nodes 100 --table time --methods sparserc golem notears
python neurips_experiments/print_table_tex.py --weight_bounds 0.5 2 --nodes 100 --omega 0.4 --table time --methods sparserc golem notears
python neurips_experiments/print_table_tex.py --sparsity 0.5 --nodes 100 --table time --methods sparserc golem notears
python neurips_experiments/print_table_tex.py --noise_std 0.1 --nodes 100 --table time --methods sparserc golem notears
python neurips_experiments/print_table_tex.py --noise gumbel --nodes 100 --table time --methods sparserc golem notears
python neurips_experiments/print_table_tex.py --transformation stand --nodes 100 --table time --methods sparserc golem notears
python neurips_experiments/print_table_tex.py --samples 100 --nodes 100  --table time --methods sparserc golem notears
python neurips_experiments/print_table_tex.py --fixSup True --table time --methods sparserc golem notears

# Plots for main paper
python neurips_experiments/plot_experiment.py --methods sparserc golem notears dagma direct_lingam pc GES --legend True
python neurips_experiments/plot_experiment.py --methods golem notears dagma direct_lingam pc GES sparserc --legend False
python neurips_experiments/plot_experiment.py --nodes 100 --samples 100 200 500 800 1000 --methods golem notears dagma pc GES sparserc

# Varying average degree plots
python neurips_experiments/plot_experiment.py --methods sparserc golem notears --legend False # first uncomment plot accuracy for varying edges
python neurips_experiments/plot_experiment.py --methods sparserc golem notears --legend True

# Comparing with LiNGAM and DirectLiNGAM
python neurips_experiments/plot_experiment.py --noise_effect spectrum --methods lingam direct_lingam sparserc --legend False # first uncomment plot accuracy for varying edges
python neurips_experiments/plot_experiment.py --noise_effect spectrum --methods sparserc lingam direct_lingam --legend True

# Tables for appendix. 
# printing SHD table in latex format
python neurips_experiments/print_table_tex.py --table SHD --nodes 100 --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --graph_type SF --nodes 100 --table SHD --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --edges 10 --nodes 100 --table SHD --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --weight_bounds 0.5 2 --nodes 100 --omega 0.4 --table SHD --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --sparsity 0.5 --nodes 100 --table SHD --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --noise_std 0.1 --nodes 100 --table SHD --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --noise gumbel --nodes 100 --table SHD --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --transformation stand --nodes 100 --table SHD --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --samples 100 --nodes 100 --table SHD --methods sparserc golem notears dagma pc GES #sortnregress ValueError
python neurips_experiments/print_table_tex.py --fixSup True --table SHD --methods sparserc golem notears dagma direct_lingam pc GES 

python neurips_experiments/print_table_tex.py --table SHD --nodes 100 --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --graph_type SF --nodes 100 --table SHD --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --edges 10 --nodes 100 --table SHD --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --weight_bounds 0.5 2 --nodes 100 --omega 0.4 --table SHD --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --sparsity 0.5 --nodes 100 --table SHD --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --noise_std 0.1 --nodes 100 --table SHD --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --noise gumbel --nodes 100 --table SHD --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --transformation stand --nodes 100 --table SHD --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --samples 100 --nodes 100 --table SHD --methods nocurl MMPC  
python neurips_experiments/print_table_tex.py --fixSup True --table SHD --methods lingam CAM nocurl FGS sortnregress MMPC

# printing TPR table in latex format
python neurips_experiments/print_table_tex.py --table TPR --nodes 100 --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --graph_type SF --nodes 100 --table TPR --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --edges 10 --nodes 100 --table TPR --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --weight_bounds 0.5 2 --nodes 100 --omega 0.4 --table TPR --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --sparsity 0.5 --nodes 100 --table TPR --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --noise_std 0.1 --nodes 100 --table TPR --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --noise gumbel --nodes 100 --table TPR --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --transformation stand --nodes 100 --table TPR --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --samples 100 --nodes 100  --table TPR --methods sparserc golem notears dagma pc GES #sortnregress ValueError
python neurips_experiments/print_table_tex.py --fixSup True --table TPR --methods sparserc golem notears dagma direct_lingam pc GES 

python neurips_experiments/print_table_tex.py --table TPR --nodes 100 --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --graph_type SF --nodes 100 --table TPR --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --edges 10 --nodes 100 --table TPR --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --weight_bounds 0.5 2 --nodes 100 --omega 0.4 --table TPR --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --sparsity 0.5 --nodes 100 --table TPR --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --noise_std 0.1 --nodes 100 --table TPR --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --noise gumbel --nodes 100 --table TPR --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --transformation stand --nodes 100 --table TPR --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --samples 100 --nodes 100  --table TPR --methods nocurl MMPC #sortnregress ValueError
python neurips_experiments/print_table_tex.py --fixSup True --table TPR --methods lingam CAM nocurl FGS sortnregress MMPC

# # printing SID table in latex format
python neurips_experiments/print_table_tex.py --table SID --nodes 100 --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --graph_type SF --nodes 100 --table SID --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --edges 10 --nodes 100 --table SID --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --weight_bounds 0.5 2 --nodes 100 --omega 0.4 --table SID --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --sparsity 0.5 --nodes 100 --table SID --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --noise_std 0.1 --nodes 100 --table SID --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --noise gumbel --nodes 100 --table SID --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --transformation stand --nodes 100 --table SID --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --samples 100 --nodes 100  --table SID --methods sparserc golem notears dagma pc GES #sortnregress ValueError
python neurips_experiments/print_table_tex.py --fixSup True --table SID --methods sparserc golem notears dagma direct_lingam pc GES 

python neurips_experiments/print_table_tex.py --table SID --nodes 100 --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --graph_type SF --nodes 100 --table SID --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --edges 10 --nodes 100 --table SID --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --weight_bounds 0.5 2 --nodes 100 --omega 0.4 --table SID --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --sparsity 0.5 --nodes 100 --table SID --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --noise_std 0.1 --nodes 100 --table SID --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --noise gumbel --nodes 100 --table SID --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --transformation stand --nodes 100 --table SID --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --samples 100 --nodes 100  --table SID --methods nocurl MMPC #sortnregress ValueError
python neurips_experiments/print_table_tex.py --fixSup True --table SID --methods lingam CAM nocurl FGS sortnregress MMPC

# # printing Time table in latex format
python neurips_experiments/print_table_tex.py --table time --nodes 100 --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --graph_type SF --nodes 100 --table time --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --edges 10 --nodes 100 --table time --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --weight_bounds 0.5 2 --nodes 100 --omega 0.4 --table time --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --sparsity 0.5 --nodes 100 --table time --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --noise_std 0.1 --nodes 100 --table time --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --noise gumbel --nodes 100 --table time --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --transformation stand --nodes 100 --table time --methods sparserc golem notears dagma direct_lingam pc GES 
python neurips_experiments/print_table_tex.py --samples 100 --nodes 100  --table time --methods sparserc golem notears dagma pc GES #sortnregress ValueError
python neurips_experiments/print_table_tex.py --fixSup True --table time --methods sparserc golem notears dagma direct_lingam pc GES 

python neurips_experiments/print_table_tex.py --table time --nodes 100 --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --graph_type SF --nodes 100 --table time --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --edges 10 --nodes 100 --table time --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --weight_bounds 0.5 2 --nodes 100 --omega 0.4 --table time --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --sparsity 0.5 --nodes 100 --table time --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --noise_std 0.1 --nodes 100 --table time --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --noise gumbel --nodes 100 --table time --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --transformation stand --nodes 100 --table time --methods lingam CAM nocurl FGS sortnregress MMPC
python neurips_experiments/print_table_tex.py --samples 100 --nodes 100  --table time --methods nocurl MMPC #sortnregress ValueError
python neurips_experiments/print_table_tex.py --fixSup True --table time --methods lingam CAM nocurl FGS sortnregress MMPC


# Plots for appendix which include all methods
python neurips_experiments/plot_experiment.py --methods sparserc golem notears dagma direct_lingam pc GES lingam CAM nocurl FGS sortnregress MMPC --legend True
python neurips_experiments/plot_experiment.py --methods golem notears dagma direct_lingam pc GES lingam CAM nocurl FGS sortnregress MMPC sparserc
python neurips_experiments/plot_experiment.py --nodes 100 --samples 100 200 500 800 1000 --methods golem notears dagma pc GES nocurl MMPC sparserc

# computing the loss of the weighted adjacency matrix apporiximation of sparserc
python neurips_experiments/norm_loss.py  --sparsity 0.05  --nodes 200  --samples 500  --runs 1 --methods sparserc 
python neurips_experiments/norm_loss.py  --sparsity 0.05  --nodes 500  --samples 1000 --runs 1 --methods sparserc 
python neurips_experiments/norm_loss.py  --sparsity 0.05  --nodes 1000 --samples 5000  --runs 1 --methods sparserc 
python neurips_experiments/norm_loss.py  --sparsity 0.05  --nodes 2000  --samples 10000  --runs 1 --methods sparserc 
python neurips_experiments/norm_loss.py  --sparsity 0.05  --nodes 3000  --samples 10000  --runs 1 --methods sparserc 
