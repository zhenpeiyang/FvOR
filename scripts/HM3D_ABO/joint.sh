#!/bin/bash -l
main_cfg=$1
EXP_NAME=exp1
n_nodes=1
n_gpus_per_node=-1
torch_num_workers=4
batch_size=4
pin_memory=true
exp_name="hm3d_abo_exp1"


PYTHONPATH="$(dirname $0)/../../":$PYTHONPATH \
python -u ./src/train_recon.py \
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
    --max_epochs=16 \
    --trainer=recon \
    --ckpt_path=./hm3d_abo_shape.ckpt \
    --dump_dir=test_results_HM3D_ABO_joint \
    ${@:2} \
    #--debug \