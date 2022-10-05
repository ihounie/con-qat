#!/bin/bash

{
    curr_dir="."
    train_id="anyp_cifar10_resnet20_"
    train_id="${train_id}_1_2_4_8_32"

    result_dir="$curr_dir/results/$train_id"
    mkdir -p $result_dir

    python -u evaluate.py \
        --model resnet20q \
        --dataset cifar10 \
        --train_split train \
        --bit_width_list "1,2,4,8,32"\
        --model_path $result_dir \
        --wandb_log
} && exit
