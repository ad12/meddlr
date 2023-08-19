#!/bin/bash
export MEDDLR_DATASETS_DIR=/mnt/dense/ozt/dl-ss-recon/data 
export MEDDLR_CACHE_DIR=/mnt/dense/deepro/cache
export MEDDLR_RESULTS_DIR=/mnt/dense/deepro/results/Summer_2022_2023/motion_eval/unet_official/ssdu/wandb

gpu=$(python get_available_gpu.py)
echo "The first available gpu is $gpu"

# Run normal unet yaml
# took off --debug to allow wb to work. 
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=$gpu$ python tools/eval_net.py --config-file /mnt/dense/deepro/results/Summer_2022_2023/unet_official/ssdu/wandb/vortex-rm/mridata_knee_3dfse/unet/ssdu/ssdu_unet_13_scan/version_001/config.yaml --metric val_psnr_mag --save-scans --angle 30 --translation 0.1 --nshots 5 --trajectory interleaved --mri_dim 2 --motion standard 


# gpu=$(python get_available_gpu.py)
# echo "The first available gpu for VORTEX is $gpu"

# # Run vortex unet yaml
# # took off --debug to allow wb to work. 
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=$gpu$ python tools/eval_net.py --config-file /mnt/dense/deepro/results/Summer_2022_2023/unet_official/ssdu/wandb/vortex-rm/mridata_knee_3dfse/unet/ssdu/ssdu_aug_unet_13_scan/version_001/config.yaml --metric val_psnr_mag --save-scans --angle 30 --translation 0.1 --nshots 5 --trajectory interleaved --mri_dim 2 --motion standard 

