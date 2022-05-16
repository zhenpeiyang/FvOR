#!/bin/bash -l

main_cfg=$1
n_nodes=1
n_gpus_per_node=-1
torch_num_workers=4
batch_size=16
pin_memory=true
exp_name="shapenet_exp1"

PYTHONPATH="$(dirname $0)/../../":$PYTHONPATH \
python -u ./src/train_pose_init.py \
    ${main_cfg} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=100 \
    --flush_logs_every_n_steps=100 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=25 \
    --trainer=recon \
    --mode=test \
    --ckpt_path=./shapenet_pose_init.ckpt\
    --save_pose_init \
    --dump_dir=test_results_pose_init \
    ${@:2} \
    #--debug \
