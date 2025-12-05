models=("resnet" "inception")
datasets=("celeba_gender" "chexpert_pleuraleffusiongender")
bias_samples_train=(25)
bias_samples_val=(10)
attenuations=(1 2 3 4)
projectroot=${PROJECTDIR}
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export WANDB_MODE=offline

for i in {0..0}; do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            for n_samples in "${!bias_samples_train[@]}"; do
                for n_attenuation in "${attenuations[@]}"; do
                    # echo "${dataset}_${model}_biased"
                    # PYTHONPATH=$projectroot python3 -u $projectroot/src/evaluation_2d_samples_exp.py --dataset $dataset \
                    #     --model $model --checkpoint  "${dataset}_${model}_biased" \
                    #     --mode eval --in-channels 3 --seed $i --bias-samples-train ${bias_samples_train[n_samples]} \
                    #     --bias-samples-val ${bias_samples_val[n_samples]} --attenuation $n_attenuation
                    
                    # echo "${dataset}_${model}_biased_masked"
                    # PYTHONPATH=$projectroot python3 -u $projectroot/src/evaluation_2d_samples_exp.py --dataset $dataset \
                    #     --model $model --checkpoint  "${dataset}_${model}_biased" \
                    #     --mode eval --in-channels 3 --seed $i --bias-samples-train ${bias_samples_train[n_samples]} \
                    #     --bias-samples-val ${bias_samples_val[n_samples]} --partition grid --regions 256 \
                    #     --masked --threshold --attenuation $n_attenuation

                    for k in {0..9}; do
                        echo "${dataset}_${model}_biased_masked_shuffle"
                            PYTHONPATH=$projectroot python3 -u $projectroot/src/evaluation_2d_samples_exp.py --dataset $dataset \
                                --model $model --checkpoint  "${dataset}_${model}_biased" \
                                --mode eval --in-channels 3 --seed $i --bias-samples-train ${bias_samples_train[n_samples]} \
                                --bias-samples-val ${bias_samples_val[n_samples]} --partition grid --regions 256 \
                                --masked --threshold --shuffle --shuffle-seed $k --attenuation $n_attenuation
                    done

                done
            done
        done
    done
done