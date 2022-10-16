#!/bin/bash
BW=2
for seed in 0 1 2
do
{
    curr_dir="."
    train_id="OURS_3l_bw_${BW}"
    result_dir="./results/$train_id"
    mkdir -p $result_dir
    python -u train_selec.py \
        --model resnet20q \
        --dataset cifar10 \
        --train_split train \
        --lr 0.001 \
        --lr_decay "50,75,90" \
        --epochs 100 \
        --optimizer adam \
        --weight-decay 0.0 \
        --results-dir $result_dir \
        --bit_width_list "${BW}" \
        --no_quant_layer "7, 8, 4" \
        --seed "${seed}" \
        --project LayerRetrain \
        --wandb_log \
        --eval_constraint
}
done
