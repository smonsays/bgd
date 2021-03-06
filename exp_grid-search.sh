#!/bin/bash
# Grid Search for two hidden-layer networks with 200 hidden units per layer,
# unpadded inputs and 10 epochs per task (seed=2019 to ensure reproducibility)

# Activate the python environment
module load gcc/6.3.0 python_gpu/3.6.4
source .myenv/bin/activate

# std_inits as suggested in readme
# std_inits=( 0.01 0.02 0.03 0.04 0.05 0.06 )
std_inits=(0.0001 0.001 0.01 0.05)
mean_etas=(0.1 1.0 10.0)

for std_init in "${std_inits[@]}"
do
	for mean_eta in "${mean_etas[@]}"
	do

	# Discrete task agnostic domain learning on Permuted MNIST (
	CUDA_VISIBLE_DEVICES=1 python main.py --logname discrete_permuted_mnist_10tasks_10epochs_std-init${std_init} --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --test_freq 10 --seed 2019 --permute_seed 2019 --dataset ds_permuted_mnist --num_epochs $(( 10 * 10 )) --optimizer bgd --std_init ${std_init} --batch_size 128 --results_dir grid-search\/perm_mnist --train_mc_iters 10 --inference_mc --test_mc_iters 10 --mean_eta ${mean_eta}

	# Discrete task agnostic domain learning on Permuted Fashion MNIST
	CUDA_VISIBLE_DEVICES=1 python main.py --logname discrete_permuted_fmnist_10tasks_10epochs_std-init${std_init} --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --test_freq 10 --seed 2019 --permute_seed 2019 --dataset ds_permuted_fmnist --num_epochs $(( 10 * 10 )) --optimizer bgd --std_init ${std_init} --batch_size 128 --results_dir grid-search\/perm_fmnist --train_mc_iters 10 --inference_mc --test_mc_iters 10 --mean_eta ${mean_eta}

	# Discrete task agnostic domain learning on Split MNIST
	CUDA_VISIBLE_DEVICES=1 python main.py --logname discrete_domain_split_mnist_5tasks_10epochs_std-init${std_init} --nn_arch mnist_simple_net_200width_domainlearning_784input_2cls_1ds --test_freq 10 --seed 2019 --permute_seed 2019 --dataset ds_split_mnist --num_epochs $(( 10 * 5 )) --optimizer bgd --std_init ${std_init} --batch_size 128 --results_dir grid-search\/split_mnist --train_mc_iters 10 --inference_mc --test_mc_iters 10 --mean_eta ${mean_eta}

	# Discrete task agnostic domain learning on Split Fashion MNIST
	CUDA_VISIBLE_DEVICES=1 python main.py --logname discrete_domain_split_fmnist_5tasks_10epochs_std-init${std_init} --nn_arch mnist_simple_net_200width_domainlearning_784input_2cls_1ds --test_freq 10 --seed 2019 --permute_seed 2019 --dataset ds_split_fmnist --num_epochs $(( 10 * 5 )) --optimizer bgd --std_init ${std_init} --batch_size 128 --results_dir grid-search\/split_fmnist --train_mc_iters 10 --inference_mc --test_mc_iters 10 --mean_eta ${mean_eta}
	done
done
