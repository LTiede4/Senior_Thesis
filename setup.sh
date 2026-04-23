#!/bin/bash

# Initialize conda for this script without needing to restart shell
eval "$(conda shell.bash hook)"

# run this script from Senior_Thesis directory, not WOFOSTGym
current_dir=$PWD
cd $current_dir/WOFOSTGym || exit

# Create conda environment with base packages
conda create -n WOFOSTGym_test python=3.12 -y
conda activate WOFOSTGym_test

# Install conda packages
cconda install -c pytorch -c nvidia -c defaults \
    pytorch=2.5.1=py3.12_cuda12.1_cudnn9.1.0_0 \
    torchvision=0.20.1=py312_cu121 \
    torchaudio=2.5.1=py312_cu121 \
    cuda-version=12.1 \
    mkl=2023.1.0 \
    intel-openmp=2023.1.0 \
    numpy=2.0.1 \
    pandas=3.0.1 \
    matplotlib=3.10.8 \
    -y

pip install tyro==1.0.8 omegaconf==2.3.0 wandb==0.25.0 tensorboard==2.20.0 tqdm==4.67.3 huggingface_sb3==3.0 gymnasium==1.0.0

pip install -e pcse -e pcse_gym
pip install -e imitation -e stable-baselines3