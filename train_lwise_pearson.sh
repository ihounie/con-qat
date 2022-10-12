#!/bin/bash
for BW in 8 4 2
do
    curr_dir="."
    train_id="pearson"
    train_id="${train_id}${BW}"
    result_dir="$curr_dir/results/$train_id"
    mkdir -p $result_dir

    python -u train_pd.py \
        --model resnet20q \
        --dataset cifar10 \
        --train_split train \
        --lr 0.001 \
        --lr_dual 0.01 \
        --lr_decay "50,75,90" \
        --epochs 100 \
        --optimizer adam \
        --weight-decay 0.0 \
        --results-dir $result_dir \
        --bit_width_list "${BW}, 32"\
        --wandb_log \
        --layerwise_constraint \
        --pearson \
        --grads_wrt_high \
        --project Pearson
done
