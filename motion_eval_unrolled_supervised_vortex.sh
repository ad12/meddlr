#!/bin/bash
export MEDDLR_DATASETS_DIR=/mnt/dense/ozt/dl-ss-recon/data 
export MEDDLR_CACHE_DIR=/mnt/dense/deepro/cache
export MEDDLR_RESULTS_DIR=/mnt/dense/deepro/results/Summer_2022_2023/motion_eval/unrolled_official/supervised/wandb

gpu=$(python get_available_gpu.py)
echo "The first available gpu is $gpu"

# took off --debug to allow wb to work. 
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=$gpu$ python tools/eval_net.py --config-file /mnt/dense/deepro/results/Summer_2022_2023/unrolled_official/supervised/wandb/vortex-rm/mridata_knee_3dfse/unrolled/supervised/supervised_unrolled_1_scan/version_001/config.yaml --metric val_psnr_mag --save-scans --angle 30 --translation 0.1 --nshots 5 --trajectory interleaved --mri_dim 2 --motion standard 

gpu=$(python get_available_gpu.py)
echo "The first available gpu is $gpu"

# took off --debug to allow wb to work. 
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=$gpu$ python tools/eval_net.py --config-file /mnt/dense/deepro/results/Summer_2022_2023/unrolled_official/supervised/wandb/vortex-rm/mridata_knee_3dfse/unrolled/supervised/supervised_unrolled_14_scan/version_001/config.yaml --metric val_psnr_mag --save-scans --angle 30 --translation 0.1 --nshots 5 --trajectory interleaved --mri_dim 2 --motion standard 

gpu=$(python get_available_gpu.py)
echo "The first available gpu for VORTEX is $gpu"

# # took off --debug to allow wb to work. 
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=$gpu$ python tools/eval_net.py --config-file /mnt/dense/deepro/results/Summer_2022_2023/unrolled_official/supervised/wandb/vortex-rm/mridata_knee_3dfse/unrolled/supervised/supervised_aug_unrolled_1_scan/version_001/config.yaml --metric val_psnr_mag --save-scans --angle 30 --translation 0.1 --nshots 5 --trajectory interleaved --mri_dim 2 --motion standard 

gpu=$(python get_available_gpu.py)
echo "The first available gpu for VORTEX is $gpu"

# # took off --debug to allow wb to work. 
WANDB__SERVICE_WAIT=300 CUDA_VISIBLE_DEVICES=$gpu$ python tools/eval_net.py --config-file /mnt/dense/deepro/results/Summer_2022_2023/unrolled_official/supervised/wandb/vortex-rm/mridata_knee_3dfse/unrolled/supervised/supervised_aug_unrolled_14_scan/version_001/config.yaml --metric val_psnr_mag --save-scans --angle 30 --translation 0.1 --nshots 5 --trajectory interleaved --mri_dim 2 --motion standard 

