models=("vit")
datasets=("chexpert_pleuraleffusiongender")
bias_samples_train=(500 1000 1500 2000)
bias_samples_val=(40 60 80 100)
methods=("LRP")
projectroot=${PROJECTDIR}

for i in {0..0}; do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            for method in "${methods[@]}"; do
                for n_samples in "${!bias_samples_train[@]}"; do
                    echo "${dataset}_${model}_biased_$method"
                    PYTHONPATH=$projectroot/src python3 -u $projectroot/src/explain_2d/vismethods.py --dataset $dataset --model $model \
                        --method $method --seed $i --bias-samples-train ${bias_samples_train[n_samples]} \
                            --bias-samples-val ${bias_samples_val[n_samples]}
                done
            done
        done
    done
done