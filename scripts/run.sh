export CUDA_VISIBLE_DEVICES=6
lrs=("1e-3")
bert_lrs=("5e-6")
model_names=("bert_rn")
input_types=("tt")

# datasets=("dt-" "hc-" "la-" "fm-")
datasets=("vast")
seeds=("2")
dropouts=("0.2")
add_nums=(1500 3000 4500 6000 7500 9000 10500 12000 13500 15000)
label_ratios=(1)
FAD_ratios=(0)
RAD_ratios=(1)
for lr in ${lrs[*]}
do
    for bert_lr in ${bert_lrs[*]}
    do
        for model_name in ${model_names[*]}
        do
            for dataset in ${datasets[*]}
            do
                for seed in ${seeds[*]}
                do
                    for dropout in ${dropouts[*]}
                    do
                        for label_ratio in ${label_ratios[*]}
                        do
                            for FAD_ratio in ${FAD_ratios[*]}
                            do
                                for RAD_ratio in ${RAD_ratios[*]}
                                do
                                    for input_type in ${input_types[*]}
                                    do
                                        for add_num in ${add_nums[*]}
                                        do
                                            python3 ../codes/train.py \
                                            --lr $lr \
                                            --model_name $model_name \
                                            --dataset $dataset \
                                            --seed $seed \
                                            --dropout $dropout \
                                            --label_ratio $label_ratio \
                                            --input_type $input_type \
                                            --FAD_ratio $FAD_ratio \
                                            --RAD_ratio $RAD_ratio \
                                            --add_num $add_num \
                                            --bert_lr $bert_lr \
                                            --num_epoch 10 \
                                            --valset_ratio 0.15 \
                                            --max_seq_len 200 \
                                            # --batch_size 64 \
                                            #  > "logs/"$learning_rate"_"$max_epoch"_"$batch_size".log" 
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# nohup bash script/PStance_bernie_run.sh > logs/_PStance_bernie.out &
