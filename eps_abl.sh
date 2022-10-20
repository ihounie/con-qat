BW=$1
for SEED in 0 1 2
do
    for EPSILON in 0.003937007874015748
    do
        train_id="${BW}_${EPSILON}"
        result_dir="./results/$train_id"
        mkdir -p $result_dir

        python -u train_pd_layers.py --seed $SEED --model resnet20q --dataset cifar10 --train_split train --lr 0.001 \
		--lr_decay "50,75,100" --epochs 100 --optimizer adam --weight-decay 0.0 --results-dir $result_dir --epsilon_out 0.0125 --lr_dual 0.01 \
            --bit_width_list "${BW}, 32" --epsilonlw $EPSILON --layerwise_constraint --wandb_log --project QSL2 --constraint_norm L2
    done
done
