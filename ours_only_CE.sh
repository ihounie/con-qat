BW=$1
for seed in 0 1 2
do
    train_id="${seed}${BW}"
    result_dir="./results/$train_id"
    mkdir -p $result_dir
    python -u train_pd_layers.py \
        --model resnet20q \
        --dataset cifar10 \
        --train_split train \
        --lr 0.001 \
        --lr_dual 0.1 \
        --lr_decay "50,75,90" \
        --epochs 100 \
        --optimizer adam \
        --weight-decay 0.0 \
        --results-dir $result_dir \
        --bit_width_list "${BW}, 32" \
        --wandb_log \
        --project QS_loss \
        --seed ${seed} \
        --epsilon_out 0.05
done
