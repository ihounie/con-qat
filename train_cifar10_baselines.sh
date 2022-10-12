#!/bin/bash
for seed in 0 1
do
for bit_width in 8 4 2 1
do
{
    curr_dir="."
    train_id="baseline100_cifar10_resnet20_${bit_width}"
    train_stamp="$(date +"%Y%m%d_%H%M%S")"
    train_id=${train_id}_${train_stamp}

    result_dir="$curr_dir/results/$train_id"
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
        --bit_width_list "${bit_width}, 32"\
        --wandb_log
        --seed "${seed}"
}
done
done
