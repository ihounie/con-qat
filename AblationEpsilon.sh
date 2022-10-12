
BW=8
for EPSILON in 0.00024509803921568627 0.0004901960784313725 0.000980392156862745 0.00196078431372549 0.00392156862745098 0.00784313725490196 0.01568627450980392 0.03137254901960784 0.06274509803921569
do
    train_id="${BW}_${EPSILON}"
    result_dir="./results/$train_id"
    mkdir -p $result_dir

    python -u train_pd.py --model resnet20q --dataset cifar10 --train_split train --lr 0.001 \
        --lr_decay "50,75,100" --epochs 100 --optimizer adam --weight-decay 0.0 --results-dir $result_dir \
        --bit_width_list "${BW}, 32" --epsilonlw $EPSILON --layerwise_constraint --wandb_log --project L2AblationEpsilon --constraint_norm L2
done