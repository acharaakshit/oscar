models=("resnet")
datasets=("ADNI")
bias_samples_percent=(0 10 20 30 40)
projectroot=${PROJECTDIR}

for i in {0..0}; do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            for n_samples_idx in "${!bias_samples_percent[@]}"; do
                echo "${dataset}_${model}_biased_${!bias_samples_percent[@]}"
                PYTHONPATH=$projectroot python3 -u $projectroot/src/probing.py --dataset $dataset \
                    --model $model --seed $i --bias-samples-percent ${bias_samples_percent[n_samples_idx]}
            done

            echo "${dataset}_${model}_baseline"
                PYTHONPATH=$projectroot python3 -u $projectroot/src/probing.py --dataset $dataset \
                    --model $model --seed $i --baseline
        done
    done
done
