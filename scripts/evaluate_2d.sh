models=("resnet" "inception")
datasets=("celeba_gender" "chexpert_pleuraleffusiongender")
projectroot=${PROJECTDIR}
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export WANDB_MODE=offline

for i in {0..0}; do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            echo "${dataset}_${model}_baseline"
            PYTHONPATH=$projectroot python3 -u $projectroot/src/evaluation_2d.py --dataset $dataset \
                --model $model --checkpoint  "${dataset}_${model}_baseline" \
                --mode eval --in-channels 3 --baseline --seed $i

            echo "${dataset}_${model}_biased"
            PYTHONPATH=$projectroot python3 -u $projectroot/src/evaluation_2d.py --dataset $dataset \
                --model $model --checkpoint  "${dataset}_${model}_biased" \
                --mode eval --in-channels 3 --seed $i 
        
            echo "${dataset}_${model}_attribute"
            PYTHONPATH=$projectroot python3 -u $projectroot/src/evaluation_2d.py --dataset $dataset \
                --model $model --checkpoint  "${dataset}_${model}_attribute" \
                --mode eval --in-channels 3 --baseline --attribute --seed $i

        done
    done
done