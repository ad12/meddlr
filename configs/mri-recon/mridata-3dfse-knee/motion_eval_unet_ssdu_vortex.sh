#!/bin/bash
export MEDDLR_DATASETS_DIR=/mnt/dense/ozt/dl-ss-recon/data 
export MEDDLR_CACHE_DIR=/mnt/dense/deepro/cache
export MEDDLR_RESULTS_DIR=/mnt/dense/deepro/results/Summer_2022_2023/motion_eval/unet_official/ssdu/wandb

gpu=$(python get_available_gpu.py)
echo "The first available gpu is $gpu"

# Run normal unet yaml
# took off --debug to allow wb to work. 
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=$gpu$ python tools/eval_net.py --config-file configs/mri-recon/mridata-3dfse-knee/unet/ssdu/SSDU_UNET_13_Scan.yaml --auto-version --save-scans --angle 30 --translation 0.1 --nshots 5 --trajectory interleaved --mri_dim 2 --motion standard MODEL.WEIGHTS 3Dmridata/SSDU/model.cpkt

# gpu=$(python get_available_gpu.py)
# echo "The first available gpu for VORTEX is $gpu"

# # Run vortex unet yaml
# # took off --debug to allow wb to work. 
# WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=$gpu$ python tools/train_net.py --config-file configs/mri-recon/mridata-3dfse-knee/unet/ssdu/SSDU_Aug_UNET_13_Scan.yaml --auto-version
