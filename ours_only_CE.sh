BW=$1
EPSILON=$2
    for seed in 3 4 5
    do
        train_id="${BW}_${seed}"
        result_dir="./results/$train_id"
        mkdir -p $result_dir
        python -u train_pd_layers.py \
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
	    --epsilon_out $2 \
            --bit_width_list "${BW}, 32" \
            --wandb_log \
            --project QS_Ours_OnlyCE \
            --seed ${seed}
    done
