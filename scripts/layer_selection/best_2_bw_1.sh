#!/bin/bash
            for seed in 0 1 2
            do
                curr_dir="."
                train_id="2l_bestd_1_${seed}"
                result_dir="./results/$train_id"
                mkdir -p $result_dir
                python -u train_pd_layers_selec.py \
                    --model resnet20q \
                    --dataset cifar10 \
                    --train_split train \
                    --lr 0.001 \
                    --lr_decay "50,75,90" \
                    --epochs 100 \
                    --optimizer adam \
                    --weight-decay 0.0 \
                    --results-dir $result_dir \
                    --bit_width_list "1" \
                    --no_quant_layer "Block_6_shortcut,Block_3_shortcut" \
                    --seed "${seed}" \
                    --wandb_log \
                    --epsilonlw 0.7 \
                    --project QSLayerRetrain
            done
            