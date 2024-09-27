seeds=(13 21)  # Specify the desired seed values
tasks=("agnews" "dbpedia" "trec" "subj" "sst-5")
templates=("*cls**sent_0*_It_is_about*mask*.*sep+*" "*cls**sent_0*_It_is_about*mask*.*sep+*" "*cls**sent_0*_It_is_about*mask*.*sep+*" "*cls**sent_0*_This_is*mask*.*sep+*" "*cls**sent_0*_The_movie_is*mask*.*sep+*")
mappings=("{'0': 'World', '1': 'Sports', '2': 'Business', '3': 'Technology'}" "{'0': 'Company', '1': 'Artist', '2': 'Building', '3': 'Nature'}" "{'0':'Number', '1':'Location', '2':'Person', '3':'Description', '4':'Entity', '5':'Expression'}" "{'0': 'objective', '1': 'subjective'}" "{'0':'terrible','1':'bad', '2':'okay', '3':'good', '4':'great'}")
class_names=('["World", "Sports", "Business", "Technology"]' '["Company", "Artist", "Building", "Nature"]' '["Number", "Location", "Person", "Description", "Entity", "Expression"]' '["objective", "subjective"]' '["v. negative", "negative", "neutral", "positive", "v. positive"]')


for i in "${!tasks[@]}"
do
    task=${tasks[i]}
    templ=${templates[i]}
    map=${mappings[i]}
    cls_name=${class_names[i]}
    echo "Task name: ${task}"
    for seed in "${seeds[@]}"
    do
        echo "Running seed ${task}_${seed}..."
        python /content/prompt_based_bias_calibration/calibrate_bias/run.py \
          --model_name_or_path roberta-large \
          --few_shot_type prompt \
          --num_k 16 \
          --num_sample 16 \
          --template ${templ} \
          --mapping "${map}" \
          --class_name "${cls_name}" \
          --write_output_file True \
          --test_mode zero_shot \
          --max_seq_length 150 \
          --task_name ${task} \
          --data_dir /content/prompt_based_bias_calibration/data/${task}/data_for_calibration/no_demo \
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


