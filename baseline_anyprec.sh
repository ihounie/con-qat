#!/bin/bash
BW=8
for seed in 0 1 2 3 4
do
{
    curr_dir="."
    train_id="anyprec_${BW}"
    result_dir="./results/$train_id"
    mkdir -p $result_dir
    python -u train.py \
        --model resnet20q \
        --dataset cifar10 \
        --train_split train \
        --lr 0.001 \
        --lr_decay "50,75,90" \
        --epochs 100 \
        --optimizer adam \
        --weight-decay 0.0 \
        --results-dir $result_dir \
        --bit_width_list "${BW},32" \
        --wandb_log \
        --seed "${seed}" \
        --project Blines \
        --eval_constraint
}
done