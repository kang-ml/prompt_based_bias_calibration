seeds=(13)  # Specify the desired seed values
tasks=("laptops" "restaurants" "tweets")

for i in "${!tasks[@]}"
do
    task=${tasks[i]}
    echo "Task name: ${task}"
    for seed in "${seeds[@]}"
    do
        echo "Running seed ${task}_${seed}..."
        python /content/prompt_based_bias_calibration/calibrate_bias/run.py \
          --model_name_or_path roberta-large \
          --few_shot_type prompt \
          --num_k 16 \
          --num_sample 16 \
          --template "*cls**sent_0**aspect_0*_was*mask*.*sep+*" \
          --mapping "{'0':'great','1':'terrible', '2':'okay'}" \
          --class_name '["positive", "negative", "neutral"]' \
          --write_output_file True \
          --test_mode zero_shot \
          --max_seq_length 150 \
          --task_name ${task} \
          --data_dir /content/prompt_based_bias_calibration/data/${task}/data_for_calibration/zero_shot \
          --save_at_first True \
          --overwrite_output_dir \
          --do_train \
          --do_predict \
          --max_steps 1 \
          --eval_steps 1 \
          --logging_steps 1 \
          --per_device_train_batch_size 32 \
          --per_device_eval_batch_size 4 \
          --learning_rate 1e-3 \
          --warmup_ratio 0.0 \
          --warmup_steps 0 \
          --weight_decay 0.0 \
          --num_train_epochs 0 \
          --output_dir /content/zero_shot_one_epoch \
          --logging_dir /content/zero_shot_one_epoch/${task}/seed_${seed}/16_shot/log \
          --save_logit \
          --seed $seed \
          --first_sent_limit 140
          echo "---------------------------------------" 
        echo "Clearing memory..."
        python -c "import gc; gc.collect()"
        echo "" > /dev/null
    done
done


