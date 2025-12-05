models=("vit")
datasets=("celeba_gender")
methods=("LRP")
projectroot=${PROJECTDIR}
for i in {0..100}; do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            for method in "${methods[@]}"; do
                echo "${dataset}_${model}_baseline_$method_$i"
                PYTHONPATH=$projectroot/src python3 -u $projectroot/src/explain_2d/vismethods.py --dataset $dataset --model $model \
                    --baseline --method $method --seed $i

                echo "${dataset}_${model}_attribute_$method$i"
                PYTHONPATH=$projectroot/src python3 -u $projectroot/src/explain_2d/vismethods.py --dataset $dataset --model $model \
                --baseline --method $method --attribute --seed $i
                
                echo "${dataset}_${model}_biased_$method_$i"
                PYTHONPATH=$projectroot/src python3 -u $projectroot/src/explain_2d/vismethods.py --dataset $dataset --model $model \
                    --method $method --seed $i
            done
        done
    done
done