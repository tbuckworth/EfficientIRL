#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tfb115

export PATH=/vol/bitbucket/${USER}/EfficientIRL/venv/bin/:/vol/cuda/12.2.0/bin/:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/vol/cuda/12.2.0/lib64:/vol/cuda/12.2.0/lib
source /vol/bitbucket/${USER}/EfficientIRL/venv/bin/activate
. /vol/cuda/12.2.0/setup.sh
TERM=vt100
/usr/bin/nvidia-smi
export CUDA_DIR=/vol/cuda/12.2.0/:${CUDAPATH}
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/vol/cuda/12.2.0/
#export MUJOCO_PATH=/vol/bitbucket/tfb115/mujoco-2.3.0
#export MUJOCO_PLUGIN_PATH=/vol/bitbucket/tfb115/mujoco-2.3.0/plugins
#pip install mujoco==2.3.0
#sleep 60
python3 /vol/bitbucket/${USER}/EfficientIRL/non_tabular.py --n_threads $1
