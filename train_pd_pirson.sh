#!/bin/bash

{
    BW=8
    curr_dir="."
    train_id="Pearson_PD_cifar10_resnet20_"
    train_id="${train_id}${BW}"

    result_dir="$curr_dir/results/$train_id"
    mkdir -p $result_dir

    python -u train_pd.py \
        --model resnet20q \
        --dataset cifar10 \
        --train_split train \
        --lr 0.001 \
        --lr_dual 0.01 \
        --lr_decay "100,150,180" \
        --epochs 200 \
        --optimizer adam \
        --weight-decay 0.0 \
        --results-dir $result_dir \
        --bit_width_list "${BW}, 32" \
        --grads_wrt_high \
        --layerwise_constraint \
        --normalise_constraint \
        --pearson \
        --wandb_log
} && exit
