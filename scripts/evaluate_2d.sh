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
            python3 -u -m evaluation_2d --dataset $dataset \
                --model $model --checkpoint  "${dataset}_${model}_baseline" \
                --mode eval --in-channels 3 --baseline --seed $i

            echo "${dataset}_${model}_biased"
            python3 -u -m evaluation_2d --dataset $dataset \
                --model $model --checkpoint  "${dataset}_${model}_biased" \
                --mode eval --in-channels 3 --seed $i 
        
            echo "${dataset}_${model}_attribute"
            python3 -u -m evaluation_2d --dataset $dataset \
                --model $model --checkpoint  "${dataset}_${model}_attribute" \
                --mode eval --in-channels 3 --baseline --attribute --seed $i

        done
    done
done