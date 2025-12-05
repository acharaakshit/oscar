models=("swin2d" "resnet")
datasets=("celeba_gender")
methods=("GradCAM")
projectroot=${PROJECTDIR}

for i in {0..0}; do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            for method in "${methods[@]}"; do
                echo "${dataset}_${model}_baseline_$method"
                PYTHONPATH=$projectroot/src python3 -u $projectroot/src/explain_2d/stab.py --dataset $dataset --model $model \
                    --baseline --method $method --seed $i

                echo "${dataset}_${model}_attribute_$method"
                PYTHONPATH=$projectroot/src python3 -u $projectroot/src/explain_2d/stab.py --dataset $dataset --model $model \
                --baseline --method $method --attribute --seed $i
                
                echo "${dataset}_${model}_biased_$method"
                PYTHONPATH=$projectroot/src python3 -u $projectroot/src/explain_2d/stab.py --dataset $dataset --model $model \
                    --method $method --seed $i
            done
        done
    done
done