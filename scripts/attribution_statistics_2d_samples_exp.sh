models=("vit")
datasets=("celeba_gender" "chexpert_pleuraleffusiongender")
# "chexpert_pleuraleffusiongender"
regions=(64 256)
partitions=("grid" "superpixel")
methods=("LRP")
bias_samples_train=(500 1000 1500 2000)
bias_samples_val=(40 60 80 100)
projectroot=${PROJECTDIR}

for i in {0..0}; do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            for partition in "${partitions[@]}"; do
                for method in "${methods[@]}"; do
                    for region in "${regions[@]}"; do
                        for n_samples in "${!bias_samples_train[@]}"; do
                            echo "${dataset}_${model}_biased_$method_$i"
                            PYTHONPATH=$projectroot/src python3 -u $projectroot/src/explain_2d/attribution_statistics_samples_exp.py \
                                --dataset $dataset --model $model --method $method --seed $i --partition_method $partition \
                                --regions $region --bias-samples-train ${bias_samples_train[n_samples]} \
                            --bias-samples-val ${bias_samples_val[n_samples]}
                        done
                    done
                done
            done
        done
    done
done