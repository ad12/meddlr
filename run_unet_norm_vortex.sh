#!/bin/bash
export MEDDLR_DATASETS_DIR=/mnt/dense/ozt/dl-ss-recon/data 
export MEDDLR_CACHE_DIR=/mnt/dense/deepro/cache
export MEDDLR_RESULTS_DIR=/mnt/dense/deepro/results/Summer_2022_2023/wandb

gpu=$(python get_available_gpu.py)
echo "The first available gpu is $gpu"

# Run normal unet yaml
# took off --debug to allow wb to work. 
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=$gpu$ python tools/train_net.py --config-file configs/mri-recon/mridata-3dfse-knee/unet.yaml --auto-version

gpu=$(python get_available_gpu.py)
echo "The first available gpu for VORTEX is $gpu"

# Run vortex unet yaml
# took off --debug to allow wb to work. 
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=$gpu$ python tools/train_net.py --config-file configs/mri-recon/mridata-3dfse-knee/vortex_rm_unet.yaml --auto-version
