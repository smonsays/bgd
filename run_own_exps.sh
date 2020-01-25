#!/bin/bash

## PERMUTED MNIST
# 20 Epochs per task: Parametrization as it actually is in the appendix
python main.py --logname discrete_permuted_mnist_10tasks_20epochs_2000width --nn_arch mnist_simple_net_2000width_domainlearning_784input_10cls_1ds --test_freq 10 --seed 2019 --permute_seed 2019 --dataset ds_permuted_mnist --num_epochs $(( 20 * 10 )) --optimizer bgd --std_init 0.015 --batch_size 256 --results_dir 20epochs --train_mc_iters 10 --inference_map

# 20 Epochs per task: Parametrization as I thought it was in the appendix
# Loss diverges for seeds {2019, 2020}
python main.py --logname discrete_permuted_mnist_10tasks_20epochs --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --test_freq 10 --seed 2020 --permute_seed 2020 --dataset ds_permuted_mnist --num_epochs $(( 20 * 10 )) --optimizer bgd --std_init 0.06 --batch_size 256 --results_dir 20epochs --train_mc_iters 10 --inference_mc --test_mc_iters 10

# Try with increased input dimension
# Loss diverges for seeds {2019}
python main.py --logname discrete_permuted_mnist_10tasks_20epochs --nn_arch mnist_simple_net_200width_domainlearning_1024input_10cls_1ds --test_freq 10 --seed 2019 --permute_seed 2019 --dataset ds_padded_permuted_mnist --num_epochs $(( 20 * 10 )) --optimizer bgd --std_init 0.06 --batch_size 256 --results_dir 20epochs --train_mc_iters 10 --inference_mc --test_mc_iters 10

# 20 Epochs per task: Parametrization as in GitHub repository for 10 epochs
# Achieves test accuracy of 82.33
python main.py --logname discrete_permuted_mnist_10tasks_20epochs --nn_arch mnist_simple_net_200width_domainlearning_1024input_10cls_1ds --test_freq 5 --seed 2019 --permute_seed 2019 --dataset ds_padded_permuted_mnist --num_epochs $(( 20 * 10 )) --optimizer bgd --std_init 0.02 --batch_size 128 --results_dir 20epochs --train_mc_iters 10 --inference_mc --test_mc_iters 10

# 10 Epochs per task (as found in official run_exps.sh)
# Achieves test accuracy of 81.88
python main.py --logname discrete_permuted_mnist_10tasks_10epochs --nn_arch mnist_simple_net_200width_domainlearning_1024input_10cls_1ds --test_freq 5 --seed 2019 --permute_seed 2019 --dataset ds_padded_permuted_mnist --num_epochs $(( 10 * 10 )) --optimizer bgd --std_init 0.02 --batch_size 128 --results_dir 10epochs --train_mc_iters 10 --inference_mc --test_mc_iters 10

# 300 Epochs per task (as found in official run_exps.sh)
# Successfully reproduces the main result of the paper (test accuracy 90.00)
python main.py --logname discrete_permuted_mnist_10tasks_300epochs --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --test_freq 100 --seed 2019 --permute_seed 2019 --dataset ds_permuted_mnist --num_epochs $(( 300 * 10 )) --optimizer bgd --std_init 0.06 --batch_size 128 --results_dir 300epochs --train_mc_iters 10 --inference_mc --test_mc_iters 10

## PERMUTED FASHION
# 300 Epochs per task
python main.py --logname discrete_permuted_fmnist_10tasks_300epochs --nn_arch mnist_simple_net_200width_domainlearning_784input_10cls_1ds --test_freq 100 --seed 2019 --permute_seed 2019 --dataset ds_permuted_fmnist --num_epochs $(( 300 * 10 )) --optimizer bgd --std_init 0.06 --batch_size 128 --results_dir permuted_fmnist --train_mc_iters 10 --inference_mc --test_mc_iters 10


## SPLIT MNIST
# Discrete task agnostic on Split MNIST (4 epochs per task) domain learning, network:[400,400] (from repo)
python main.py --logname discrete_domain_split_mnist_5tasks_4epochs_seed2019 --nn_arch mnist_simple_net_400width_domainlearning_1024input_2cls_1ds --test_freq 5 --seed 2019 --permute_seed 2019 --dataset ds_padded_split_mnist --num_epochs $(( 5 * 4 )) --optimizer bgd --std_init 0.02 --batch_size 128 --results_dir split_mnist --train_mc_iters 10 --inference_mc --test_mc_iters 10

# Discrete task agnostic domain learning on Split MNIST, 10 epochs, no padding , network:[200,200]
python main.py --logname discrete_domain_split_mnist_5tasks_10epochs_seed2019 --nn_arch mnist_simple_net_200width_domainlearning_784input_2cls_1ds --test_freq 5 --seed 2019 --permute_seed 2019 --dataset ds_split_mnist --num_epochs $(( 10 * 5 )) --optimizer bgd --std_init 0.06 --batch_size 128 --results_dir split_mnist --train_mc_iters 10 --inference_mc --test_mc_iters 10


## SPLIT FASHIONMNIST
# Discrete task agnostic on Split Fashion MNIST (5 epochs per task) domain learning, network:[400,400]
python main.py --logname discrete_domain_split_fmnist_5tasks_4epochs_seed2019 --nn_arch mnist_simple_net_400width_domainlearning_1024input_2cls_1ds --test_freq 5 --seed 2019 --permute_seed 2019 --dataset ds_padded_split_fmnist --num_epochs $(( 5 * 4 )) --optimizer bgd --std_init 0.02 --batch_size 128 --results_dir split_fmnist --train_mc_iters 10 --inference_mc --test_mc_iters 10

