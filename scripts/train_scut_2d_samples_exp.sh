models=("vit")
datasets=("chexpert_pleuraleffusiongender")
bias_samples_train=(500 1000 1500 2000)
bias_samples_val=(40 60 80 100)
projectroot=${PROJECTDIR}
max_epochs=600
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export WANDB_MODE=offline

for i in {0..0}; do
    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            for n_samples in "${!bias_samples_train[@]}"; do
                echo "$projectroot,$model, $dataset, biased, $i"
                    PYTHONPATH=$projectroot python3 -u $projectroot/src/train_2d_samples_exp.py --dataset $dataset --model $model \
                        --max-epochs $max_epochs \
                        --in-channels 3 --batch-size 32 --seed $i --bias-samples-train ${bias_samples_train[n_samples]} \
                        --bias-samples-val ${bias_samples_val[n_samples]}
                
                
            done
        done
    done
done