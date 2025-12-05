models=("resnet")
datasets=("ADNI")
projectroot=${PROJECTDIR}
bias_samples_percent=(10 20 30 40)
max_epochs=150
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export WANDB_MODE=offline

for i in {0..0}; do
    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            # for n_samples_idx in "${!bias_samples_percent[@]}"; do
                echo "$projectroot,$model, $dataset, $task, baseline"
                PYTHONPATH=$projectroot python3 -u $projectroot/src/train.py --dataset $dataset --model $model \
                    --max-epochs $max_epochs --in-channels 1 --baseline --seed $i
                
                echo "$projectroot,$model, $dataset, $task, biased"
                PYTHONPATH=$projectroot python3 -u $projectroot/src/train.py --dataset $dataset --model $model \
                    --max-epochs $max_epochs --in-channels 1 --seed $i #--bias-samples-percent ${bias_samples_percent[n_samples_idx]}
                
                echo "$projectroot,$model, $dataset, $task, attribute"
                PYTHONPATH=$projectroot python3 -u $projectroot/src/train.py --dataset $dataset --model $model \
                    --max-epochs $max_epochs --in-channels 1 --baseline --attribute --seed $i
            # done
        done
    done
done