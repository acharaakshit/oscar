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
                        python3 -u -m explain.attribution_statistics \
                            --dataset $dataset --model $model \
                            --baseline --seed $i --partition $partition --regions $region

                        echo "${dataset}_${model}_biased"
                        python3 -u -m explain.attribution_statistics \
                            --dataset $dataset --model $model \
                            --seed $i --partition $partition --regions $region 
                            # \
                            # --bias-samples-percent ${bias_samples_percent[n_samples_idx]}

                        echo "${dataset}_${model}_attribute"
                        python3 -u -m explain.attribution_statistics \
                            --dataset $dataset --model $model \
                            --baseline --attribute --seed $i --partition $partition --regions $region
                    done
                done
            done
        done
    done
done