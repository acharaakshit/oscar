models=("resnet")
datasets=("ADNI")
bias_samples_percent=(10 20 30 40)
projectroot=${PROJECTDIR}
for i in {0..0}; do
    for model in "${models[@]}"; do
        for dataset in "${datasets[@]}"; do
            # for n_samples_idx in "${!bias_samples_percent[@]}"; do
                echo "${dataset}_${model}_baseline"
                python3 -u -m explain.save_attributions --dataset $dataset --model $model \
                    --baseline --seed $i
                
                echo "${dataset}_${model}_biased_${bias_samples_percent[n_samples_idx]}"
                python3 -u -m explain.save_attributions --dataset $dataset --model $model \
                    --seed $i #--bias-samples-percent ${bias_samples_percent[n_samples_idx]}
                
                echo "${dataset}_${model}_attribute"
                python3 -u -m explain.save_attributions --dataset $dataset --model $model \
                    --baseline --attribute --seed $i
            # done
        done
    done
done