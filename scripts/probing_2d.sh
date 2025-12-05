models=("resnet" "inception")
datasets=("celeba_gender" "chexpert_pleuraleffusiongender")
bias_samples_train=(25 500 1000 1500 2000)
bias_samples_val=(10 40 60 80 100)
projectroot=${PROJECTDIR}
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export WANDB_MODE=offline

for i in {0..0}; do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            echo "${dataset}_${model}_baseline"
            PYTHONPATH=$projectroot python3 -u $projectroot/src/probing_2d.py --dataset $dataset \
                --model $model --in-channels 3 --seed $i --baseline

            for n_samples in "${!bias_samples_train[@]}"; do
                echo "${dataset}_${model}_biased_${bias_samples_train[n_samples]}"
                PYTHONPATH=$projectroot python3 -u $projectroot/src/probing_2d.py --dataset $dataset \
                    --model $model --in-channels 3 --seed $i --bias-samples-train ${bias_samples_train[n_samples]} \
                            --bias-samples-val ${bias_samples_val[n_samples]}
            done
        done
    done
done