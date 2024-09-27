# Enhancing Zero/Few-Shot Learning of Pre-trained Language Models via Bias Calibration
This is the implementation of the paper [Prompt-Based Bias Calibration for Better Zero/Few-Shot Learning of Language Models](https://arxiv.org/abs/2402.10353)
(EMNLP 2024 Findings). Code is modified from [LM-BFF](https://github.com/princeton-nlp/LM-BFF).

## Overview
We propose a null-input prompting method for calibrating intrinsic bias of pre-trained language models, aiming to enhance language models’ zero/few-shot performance for both in-context learning and prompt-based fine-tuning. Here are some main contributions:
- A new perspective of improving pre-trained LM zero/few-shot learning via _intrinsic-bias calibration_.
- Unsupervised intrinsic bias calibration using KL-divergence loss.
- Efficiency and minimal model updates.
  
![overview](overview.png)

## Instructions
### Data
The `data` folder includes all the eight datasets in the experiment. 
- `data/agnews` `data/dbpedia` `data/sst-5` `data/trec` `data/subj` are sentence-level classification task.
- `data/restaurants` `data/laptops` `data/tweets` are aspect-level classification tasks.
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
