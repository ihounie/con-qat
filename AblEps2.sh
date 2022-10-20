
BW=2
EPSILONOUT2=0.7
for seed in 0 1 2
do
    for EPSILONLW in 0.99999 1.5 
    do
        train_id="${BW}_${EPSILONLW}_${seed}"
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
            --bit_width_list "${BW}, 32" \
            --wandb_log \
            --epsilonlw $EPSILONLW \
            --layerwise_constraint \
            --constraint_norm L2 \
            --epsilon_out $EPSILONOUT2 \
            --project QS_L2AblEps_CEconstraint \
            --seed ${seed}
    done
done