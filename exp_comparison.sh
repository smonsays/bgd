#!/bin/bash
# Main experiments for two hidden-layer networks with 200 hidden units per layer,
# unpadded inputs, 10 epochs per task and optimal parameters found in grid search

# Activate the virtual python environment
module load gcc/6.3.0 python_gpu/3.6.4
source .myenv/bin/activate

seeds=( 2019 2020 2021 )

for seed in "${seeds[@]}"
do
	# Discrete task agnostic on Permuted Fashion MNIST (10 epochs per task) - domain learning
	CUDA_VISIBLE_DEVICES=1 python main.py --logname discrete_permuted_fmnist_10tasks_10epochs_seed${seed} --nn_arch mnist_simple_net_200width_domainlearning_1024input_10cls_1ds --test_freq 10 --seed ${seed} --permute_seed ${seed} --dataset ds_padded_permuted_fmnist --num_epochs $(( 10 * 10 )) --optimizer bgd --std_init 0.01 --batch_size 128 --results_dir comparison\/perm_fmnist --train_mc_iters 10 --inference_mc --test_mc_iters 10 &

	# Discrete task agnostic on Permuted MNIST (10 epochs per task) - domain learning
	CUDA_VISIBLE_DEVICES=1 python main.py --logname discrete_permuted_mnist_10tasks_10epochs_seed${seed} --nn_arch mnist_simple_net_200width_domainlearning_1024input_10cls_1ds --test_freq 10 --seed ${seed} --permute_seed ${seed} --dataset ds_padded_permuted_mnist --num_epochs $(( 10 * 10 )) --optimizer bgd --std_init 0.05 --batch_size 128 --results_dir comparison\/perm_mnist --train_mc_iters 10 --inference_mc --test_mc_iters 10 &
	
	# Discrete task agnostic on Split MNIST (10 epochs per task) domain learning
	CUDA_VISIBLE_DEVICES=1 python main.py --logname discrete_domain_split_mnist_5tasks_10epochs_seed${seed} --nn_arch mnist_simple_net_400width_domainlearning_1024input_2cls_1ds --test_freq 10 --seed ${seed} --permute_seed ${seed} --dataset ds_padded_split_mnist --num_epochs $(( 5 * 10 )) --optimizer bgd --std_init 0.01 --batch_size 128 --results_dir comparison\/split_mnist --train_mc_iters 10 --inference_mc --test_mc_iters 10 &

	# Discrete task agnostic on Split fashion MNIST (10 epochs per task) domain learning
	CUDA_VISIBLE_DEVICES=1 python main.py --logname discrete_domain_split_fmnist_5tasks_10epochs_seed${seed} --nn_arch mnist_simple_net_400width_domainlearning_1024input_2cls_1ds --test_freq 10 --seed ${seed} --permute_seed ${seed} --dataset ds_padded_split_fmnist --num_epochs $(( 5 * 10 )) --optimizer bgd --std_init 0.01 --batch_size 128 --results_dir comparison\/split_fmnist --train_mc_iters 10 --inference_mc --test_mc_iters 10 &

	# Only run 4 processes in parallel
	wait
done
