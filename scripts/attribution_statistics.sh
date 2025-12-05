models=("resnet")
datasets=("ADNI")
regions=(512)
bias_samples_percent=(10 20 30 40)
partitions=("superpixel")
projectroot=${PROJECTDIR}

for i in {0..0}; do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            for partition in "${partitions[@]}"; do
                for region in "${regions[@]}"; do
                    for n_samples_idx in "${!bias_samples_percent[@]}"; do
                        echo "${dataset}_${model}_baseline"
                        PYTHONPATH=$projectroot/src python3 -u $projectroot/src/explain/attribution_statistics.py \
                            --dataset $dataset --model $model \
                            --baseline --seed $i --partition $partition --regions $region

                        echo "${dataset}_${model}_biased"
                        PYTHONPATH=$projectroot/src python3 -u $projectroot/src/explain/attribution_statistics.py \
                            --dataset $dataset --model $model \
                            --seed $i --partition $partition --regions $region 
                            # \
                            # --bias-samples-percent ${bias_samples_percent[n_samples_idx]}

                        echo "${dataset}_${model}_attribute"
                        PYTHONPATH=$projectroot/src python3 -u $projectroot/src/explain/attribution_statistics.py \
                            --dataset $dataset --model $model \
                            --baseline --attribute --seed $i --partition $partition --regions $region
                    done
                done
            done
        done
    done
done