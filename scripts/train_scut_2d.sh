models=("vit")
datasets=("chexpert_pleuraleffusiongender")
projectroot=${PROJECTDIR}
max_epochs=600
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export WANDB_MODE=offline

for i in {0..0}; do
    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            echo "$projectroot,$model, $dataset, biased, $i"
                PYTHONPATH=$projectroot python3 -u $projectroot/src/train_2d.py --dataset $dataset --model $model \
                    --max-epochs $max_epochs \
                    --in-channels 3 --batch-size 32 --seed $i

            echo "$projectroot,$model, $dataset, baseline, $i"
                PYTHONPATH=$projectroot python3 -u $projectroot/src/train_2d.py --dataset $dataset --model $model \
                    --max-epochs $max_epochs \
                    --in-channels 3 --baseline --batch-size 32 --seed $i
            
            echo "$projectroot,$model, $dataset, baseline, attribute, $i"
                PYTHONPATH=$projectroot python3 -u $projectroot/src/train_2d.py --dataset $dataset --model $model \
                    --max-epochs $max_epochs \
                    --in-channels 3 --baseline --batch-size 32 --attribute --seed $i
        done
    done
done