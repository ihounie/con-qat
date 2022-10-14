BW=8
for seed in 0 1 2
do
    train_id="layer_norm_const"
    train_id="${train_id}${BW}"
    result_dir="./results/$train_id"
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
        --bit_width_list "${BW}, 32" \
        --grads_wrt_high \
        --wandb_log \
        --project Ours_OnlyCE \
        --seed ${seed}
done