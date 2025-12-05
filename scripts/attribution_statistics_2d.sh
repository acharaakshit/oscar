models=("vit")
datasets=("celeba_gender" "chexpert_pleuraleffusiongender")
# "chexpert_pleuraleffusiongender"
regions=(64 256)
partitions=("grid" "superpixel")
methods=("LRP")
attenuations=(1 2 3 4)
projectroot=${PROJECTDIR}


for i in {0..0}; do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            for partition in "${partitions[@]}"; do
                for method in "${methods[@]}"; do
                    for region in "${regions[@]}"; do
                        # for n_attenuation in "${attenuations[@]}"; do
                            echo "${dataset}_${model}_baseline_$method_$i"
                            PYTHONPATH=$projectroot/src python3 -u $projectroot/src/explain_2d/attribution_statistics.py \
                                --dataset $dataset --model $model --baseline --method $method --seed $i --partition_method $partition \
                                --regions $region  #--attenuation $n_attenuation
                            
                            echo "${dataset}_${model}_attribute_$method_$i"
                            PYTHONPATH=$projectroot/src python3 -u $projectroot/src/explain_2d/attribution_statistics.py \
                                --dataset $dataset --model $model --baseline --attribute --method $method --seed $i --partition_method $partition \
                                --regions $region #--attenuation $n_attenuation
                            
                            echo "${dataset}_${model}_biased_$method_$i"
                            PYTHONPATH=$projectroot/src python3 -u $projectroot/src/explain_2d/attribution_statistics.py \
                                --dataset $dataset --model $model --method $method --seed $i --partition_method $partition \
                                --regions $region #--attenuation $n_attenuation
                        # done
                    done
                done
            done
        done
    done
done