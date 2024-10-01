# Enhancing Zero/Few-Shot Learning of Pre-trained Language Models via Bias Calibration
This is the implementation of the paper [Prompt-Based Bias Calibration for Better Zero/Few-Shot Learning of Language Models](https://arxiv.org/abs/2402.10353)
(EMNLP 2024 Findings). Code is modified from [LM-BFF](https://github.com/princeton-nlp/LM-BFF).

## Overview
We propose a null-input prompting method for calibrating intrinsic bias of pre-trained language models, aiming to enhance language models’ zero/few-shot performance for both in-context learning and prompt-based fine-tuning. Here are some main contributions:
- A new perspective of improving pre-trained LM zero/few-shot learning via _intrinsic-bias calibration_.
- Unsupervised intrinsic bias calibration using KL-divergence loss.
- Efficiency and minimal model updates.
  
![overview](overview.png)

## Data
The `data` folder includes all the eight datasets in the experiment. 
- `data/agnews` `data/dbpedia` `data/sst-5` `data/trec` `data/subj` are sentence-level classification task.
- `data/restaurants` `data/laptops` `data/tweets` are aspect-level classification task.

+ `data/.../data_for_calibration` is used for bias calibration.
  
  + `.../no_demo`: The inputs to the model don't include few-shot demonstrations. This calibration is for **zero-shot** or **few-shot without demonstration** dowmstream tasks.
    + `.../train.json`: This file contains 32 null-meaning inputs for _One-batch Calibration_ as early-stopping (Section 3.3 in the paper). The `label` for these null inputs are random and not used in calibration. Please ignore.
    + `.../support.json`: This file is not used in this `no_demo` case.
    + `.../dev.json`: This file is not used in this `no_demo` case.
    + `.../test.json`: This file is the test set for the dataset.
   
  + `.../with_demo`: The inputs to the model include few-shot demonstrations. This calibration is for **few-shot with demonstration** dowmstream tasks.
    + `.../train.json`: This file contains 800 null-meaning inputs. In this case, we don't use _One-batch Calibration_ (Section 3.3 in the paper). The `label` for these null inputs are random and not used in calibration. Please ignore.
    + `.../support.json`: This file is used to provide demonstrations for the input.
    + `.../dev.json`: In this case, some labeled data are available. We use these labeled data to decide the early-stopping point of calibration.
    + `.../test.json`: This file is the test set for the dataset.
      
+ `data/.../data_for_16_shot_learning` is for prompt-based fine-tuning. We use five different random seeds to selected `train.json` and `dev.json`. `test.json` is the same for the same dataset.

## Code Running
### Requirement
To run our code, please install all the packages by using the following command:
```
pip install -r requirement.txt
```

### Run bias calibration
The following scripts is an example for `AGNews` dataset.

```
python /content/prompt_based_bias_calibration/calibrate_bias/run.py \
  --model_name_or_path roberta-large \
  --few_shot_type prompt \
  --num_k 16 \
  --num_sample 16 \
  --template "*cls**sent_0*_It_is_about*mask*.*sep+*" \
  --mapping "{'0': 'World', '1': 'Sports', '2': 'Business', '3': 'Technology'}" \
  --class_name '["World", "Sports", "Business", "Technology"]' \
  --write_output_file True \
  --test_mode zero_shot \
  --max_seq_length 150 \
  --task_name "agnews" \
  --data_dir /content/prompt_based_bias_calibration/data/agnews/data_for_calibration/no_demo \
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
  --logging_dir /content/zero_shot_one_epoch/agnews/seed_13/16_shot/log \
  --save_logit \
  --seed 13 \
  --first_sent_limit 140
```
Most arguments are inherited from `transformers` and [LM-BFF](https://github.com/princeton-nlp/LM-BFF). We provide some explanations in the following:

+ `model_name_or_path`: this part of code only applied to RoBERTa model and its fine-tuned versions.
+ `few_shot_type`:
  + `= prompt`: The inputs to the model don't include few-shot demonstrations.
  + `= prompt-demo`: The inputs to the model include few-shot demonstrations.
+ `num_k`: For indexing logs and output directories.
+ `num_sample`: for `few_shot_type: prompt-demo`, we sample `num_sample` different sets of demonstrations for one input, and average the logits for all `num_sample` samples.
