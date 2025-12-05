models=("swintransformer" "resnet")
datasets=("ADNI" "OASIS" "A4" "HCP")
tasks=("gender")
augment=true
projectroot=${PROJECTDIR}
max_epochs=600

for task in "${tasks[@]}"; do
    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            echo "$projectroot,$model, $dataset, $task, $augment, 3T"
            PYTHONPATH=$projectroot python3 -u $projectroot/src/train.py --dataset $dataset --model $model \
                --max-epochs $max_epochs --task $task \
                --augment --in-channels 1
        done
    done
done