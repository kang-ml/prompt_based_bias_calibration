"""Finetuning the library models for sequence classification on GLUE."""

import dataclasses
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import HfArgumentParser, TrainingArguments, set_seed

from src.dataset import FewShotDataset
from src.models import RobertaForPromptFinetuning
from src.trainer import Trainer
from src.processors import processors_mapping, num_labels_mapping, output_modes_mapping, compute_metrics_mapping, bound_mapping

from filelock import FileLock
from datetime import datetime

from copy import deepcopy
from tqdm import tqdm
import json

import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    # Few-shot type
    #   - finetune: standard fine-tuning
    #   - prompt: prompt-based fine-tuning
    #   - prompt-demo: prompt-based fine-tuning with demonstrations
    few_shot_type: str = field(
        default='prompt-demo',
        metadata={"help": "Few-shot learning model type. Choice: finetune, prompt, prompt-demo"}
    )


@dataclass
class DynamicDataTrainingArguments(DataTrainingArguments):
    """
    Arguments for dynamic training.
    """
    num_k: Optional[int] = field(
        default=16,
        metadata={"help": "For indexing logs and output directories"}
    )

    num_sample: Optional[int] = field(
        default=16,
        metadata={"help": "Number of samples (for inference) in fine-tuning with demonstrations"}
    )

    num_demo: Optional[int] = field(
        default=1,
        metadata={"help": "Number of demonstrations from each class"}
    )

    auto_demo: bool = field(
        default=True,
        metadata={"help": "Automatically generate template for using demonstrations"}
    )

    # For prompting
    template: str = field(
        default=None,
        metadata={"help": "Template"}
    )

    mapping: str = field(
        default=None,
        metadata={"help": "Label word mapping"}
    )
    
    ########################################
    class_name: str = field(
        default='["positive", "negative", "neutral"]', 
        metadata={"help": "Class name"}
    )

    write_output_file: bool = field(
        default=True,
        metadata={"help": "If write predictions of test data"}
    )

    test_mode: str = field(
        default=None, 
        metadata={"help": "Detect bias/zero-shot/in-context few-shot/fine-tune"}
    )

    ########################################

    template_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used"}
    )

    mapping_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used"}
    )

    prompt_path: str = field(
        default=None,
        metadata={"help": "Path to a txt file that stores all the prompts (templates and mappings), one per line"}
    )
 
    template_id: int = field(
        default=None,
        metadata={"help": "Template id if using template_path"}
    )

    mapping_id: int = field(
        default=None,
        metadata={"help": "Mapping id if using template_path"}
    )

    prompt_id: int = field(
        default=None,
        metadata={"help": "Prompt id if using prompt_path"}
    )

    top_n_template: int = field(
        default=None,
        metadata={"help": "Use top-n template in the template path"}
    )

    # For logging
    tag: str = field(
        default='',
        metadata={"help": "Set the tag and find the result easier in the log."}
    )

    # For filtering when using demonstrations
    demo_filter: bool = field(
        default=False,
        metadata={"help": "Only use similar instances in demonstrations"}
    )

    demo_filter_rate: float = field(
        default=0.5,
        metadata={"help": "Only use top-x\% similar instances in demonstrations"}
    )

    demo_filter_model: str = field(
        default=None,
        metadata={"help": "Model name for demonstration filter embeddings. Will load embeddings based on the model name."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug mode"}
    )

    # For max length
    double_demo: bool = field(
        default=False,
        metadata={"help": "Use double length for using demonstrations"}
    )

    first_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of the first sentence (i.e., sent_0)"}
    )

    other_sent_limit: int = field(
        default=None,
        metadata={"help": "Limit the length of sentences other than the first sentence"}
    )

    use_full_length: bool = field(
        default=None,
        metadata={"help": "Use the full length (512)"}
    )

    truncate_head: bool = field(
        default=False,
        metadata={"help": "When exceeding the maximum length, truncate the head instead of the tail."}
    )

    # Do not set up the following fields. They are set up automatically.
    prompt: bool = field(
        default=False,
        metadata={"help": "Whether to use prompt-based fine-tuning"}
    )



@dataclass
class DynamicTrainingArguments(TrainingArguments):
    # For ensemble
    array_id: int = field(
        default=-1,
        metadata={"help": "Array ID (contains seed and hyper-paramter search) to idenfity the model"}
    )

    model_id: int = field(
        default=-1,
        metadata={"help": "Model ID (contains template information) to identify the model"}
    )

    save_logit: bool = field(
        default=False,
        metadata={"help": "Save test file logit with name $TASK-$MODEL_ID-$ARRAY_ID.npy"}
    )

    save_logit_dir: str = field(
        default=None,
        metadata={"help": "Where to save the prediction result"}
    )

    # Regularization
    fix_layers: int = field(
        default=0,
        metadata={"help": "Fix bottom-n layers when optimizing"}
    )

    # Training
    ###############################################
    save_at_first: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the first checkpoint"}
    )
    ###############################################
    
    save_at_last: bool = field(
        default=False,
        metadata={"help": "Instead of saving the best (dev performance) checkpoint, save the last checkpoint"}
    )

    # Turn off train/test
    no_train: bool = field(
        default=False,
        metadata={"help": "No training"}
    )
    no_predict: bool = field(
        default=False,
        metadata={"help": "No test"}
    )


def main():
    parser = HfArgumentParser((ModelArguments, DynamicDataTrainingArguments, DynamicTrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if 'prompt' in model_args.few_shot_type:
        data_args.prompt = True

    if training_args.no_train:
        training_args.do_train = False
    if training_args.no_predict:
        training_args.do_predict = False

    ###############
    template_list = None
    ###############

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    # Load prompt/template/mapping file
    if data_args.prompt:
        if data_args.prompt_path is not None:
            assert data_args.prompt_id is not None
            prompt_list = []
            with open(data_args.prompt_path) as f:
                for line in f:
                    line = line.strip()
                    template, mapping = line.split('\t')
                    prompt_list.append((template, mapping))

            data_args.template, data_args.mapping = prompt_list[data_args.prompt_id] 
            logger.info("Specify load the %d-th prompt: %s | %s" % (data_args.prompt_id, data_args.template, data_args.mapping))
        else:
            if data_args.template_path is not None:
                with open(data_args.template_path) as f:
                    template_list = []
                    for line in f:
                        line = line.strip()
                        if len(line) > 0:
                            template_list.append(line)

                # Load top-n templates
                if data_args.top_n_template is not None:
                    template_list = template_list[:data_args.top_n_template]
                logger.info("Load top-%d templates from %s" % (len(template_list), data_args.template_path))

                # ... or load i-th template
                if data_args.template_id is not None:
                    data_args.template = template_list[data_args.template_id]
                    template_list = None
                    logger.info("Specify load the %d-th template: %s" % (data_args.template_id, data_args.template))

            if data_args.mapping_path is not None:
                assert data_args.mapping_id is not None # Only can use one label word mapping
                with open(data_args.mapping_path) as f:
                    mapping_list = []
                    for line in f:
                        line = line.strip()
                        mapping_list.append(line)

                data_args.mapping = mapping_list[data_args.mapping_id]
                logger.info("Specify using the %d-th mapping: %s" % (data_args.mapping_id, data_args.mapping))

    # Check save path
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists.")

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Data parameters %s", data_args)
    # Set seed
    set_seed(training_args.seed)

    try:
        num_labels = num_labels_mapping[data_args.task_name]  # =3
        output_mode = output_modes_mapping[data_args.task_name] # ="classification"
        logger.info("Task name: {}, number of labels: {}, output mode: {}".format(data_args.task_name, num_labels, output_mode))
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Automatically generate template for using demonstrations
    if data_args.auto_demo and model_args.few_shot_type == 'prompt-demo':
        logger.info("Automatically convert the template to using demonstrations.")
        if template_list is not None:
            for i in range(len(template_list)):
                old_template = template_list[i]
                new_template = old_template + ''
                old_template = old_template.replace('*cls*', '')
                # Single sentence or sentence pair?
                sent_num = 1
                if "_1" in old_template:
                    sent_num = 2
                for label_id in range(num_labels):
                    sub_template = old_template + ''
                    # Replace sent id
                    for sent_id in range(sent_num):
                        sub_template = sub_template.replace("_{}*".format(sent_id), "_{}*".format(sent_num + sent_num * label_id + sent_id))
                    # Replace mask
                    sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                    new_template = new_template + sub_template
                logger.info("| {} => {}".format(template_list[i], new_template))
                template_list[i] = new_template
        else:
            old_template = data_args.template
            new_template = old_template + ''
            old_template = old_template.replace('*cls*', '')
            # Single sentence or sentence pair?
            sent_num = 1
            # "*cls**sent_0*_It_was*mask*.*sep+*"
            if "_1" in old_template:
                sent_num = 2
            for label_id in range(num_labels):
                sub_template = old_template + ''
                # Replace sent id
                for sent_id in range(sent_num):
                    sub_template = sub_template.replace('sent', '+sent')  ##################
                    sub_template = sub_template.replace("_{}".format(sent_id), "_{}".format(sent_num + sent_num * label_id + sent_id))
                # Replace mask
                sub_template = sub_template.replace("*mask*", "*label_{}*".format(label_id))
                new_template = new_template + sub_template
            logger.info("| {} => {}".format(data_args.template, new_template))
            data_args.template = new_template

    # Create config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )

    if 'prompt' in model_args.few_shot_type:
        if config.model_type == 'roberta':
            model_fn = RobertaForPromptFinetuning
        else:
            raise NotImplementedError
    elif model_args.few_shot_type == 'finetune':
        model_fn = AutoModelForSequenceClassification
    else:
        raise NotImplementedError
    special_tokens = []

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        additional_special_tokens=special_tokens,
        cache_dir=model_args.cache_dir,
    )

    ########################################
    # get maximum input tokens length
    def get_max_length(dataset):
        max_length = 0
        for data in dataset:
            length = data.input_ids.index(tokenizer.pad_token_id)
            max_length = max(max_length, length)
        return max_length
    ########################################

    # Get our special datasets.
    train_dataset = (
        FewShotDataset(data_args, template_list, tokenizer=tokenizer, mode="train", use_demo=("demo" in model_args.few_shot_type))
    )
   
    eval_dataset = (
        FewShotDataset(data_args, template_list, tokenizer=tokenizer, mode="dev", use_demo=("demo" in model_args.few_shot_type))
    )
    

    test_dataset = (
        FewShotDataset(data_args, template_list, tokenizer=tokenizer, mode="test", use_demo=("demo" in model_args.few_shot_type))
    )

  
    set_seed(training_args.seed)

    model = model_fn.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Pass dataset and argument information to the model
    if data_args.prompt:
        model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
    if output_modes_mapping[data_args.task_name] == 'regression':
        # lower / upper bounds
        model.lb, model.ub = bound_mapping[data_args.task_name]
    model.model_args = model_args
    model.data_args = data_args
    model.tokenizer = tokenizer

    # Build metric
    def build_compute_metrics_fn(task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            # Note: the eval dataloader is sequential, so the examples are in order.
            # We average the logits over each sample for using demonstrations.
            predictions = p.predictions
            num_logits = predictions.shape[-1]
            logits = predictions.reshape([test_dataset.num_sample, -1, num_logits])
            logits = logits.mean(axis=0)
            

            if num_logits == 1:
                preds = np.squeeze(logits)
            else:
                preds = np.argmax(logits, axis=1)

            # Just for sanity, assert label ids are the same.
            label_ids = p.label_ids.reshape([eval_dataset.num_sample, -1])
            label_ids_avg = label_ids.mean(axis=0)
            label_ids_avg = label_ids_avg.astype(p.label_ids.dtype)
            assert (label_ids_avg - label_ids[0]).mean() < 1e-2
            label_ids = label_ids[0]

            return compute_metrics_mapping[task_name](task_name, preds, label_ids)  # return {"acc": (preds == labels).mean()}

        return compute_metrics_fn

    #########################################################################
    # analyze confusion matrix
    def show_confusion_matrix(confusion_matrix):
        hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('Truth')
        plt.xlabel('Predictions')
        
        ###################################################
        output_path = os.path.join(training_args.output_dir, 'confusion_matrix')
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        output_figname = os.path.join(output_path, f"{data_args.test_mode}.png") 
        plt.savefig(output_figname)
        ###################################################
        plt.show()

    def confusion_matrix_analysis(num_samples, output, write_output_file=False):
        class_names = json.loads(data_args.class_name) # ['positive', 'negative', 'neutral']

        predictions = output.predictions
        num_logits = predictions.shape[-1]
        logits = predictions.reshape([num_samples, -1, num_logits]).mean(axis=0)
        # assert logits.shape == (data_args.num_sample * (len(data_args.mapping)), len(data_args.mapping))
        preds = np.argmax(logits, axis=1)
        label_ids = output.label_ids.reshape([num_samples, -1])[0]
        
        #########################################
        if write_output_file:
           write_predict_test_data_results(label_ids, preds, class_names, data_args.data_dir, training_args.output_dir) 
        #########################################

        #########################################
        # show and save confusion matrix
        cm = confusion_matrix(label_ids, preds)
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        show_confusion_matrix(df_cm)
        #########################################


        #########################################
        # report data (F1, accuracy)
        report_data = classification_report(label_ids, preds, target_names=class_names, digits=4)

        report_path = os.path.join(training_args.output_dir, 'report_f1_acc')
        if not os.path.isdir(report_path):
            os.mkdir(report_path)    
        report_path = os.path.join(report_path, f"{data_args.test_mode}_report.txt") 

        with open(report_path, "w") as report_file:
            report_file.write(report_data)

        print(report_data)
        print(f"Classification report saved to {report_path}")
        #########################################

        return (preds == label_ids).mean()

    def write_predict_test_data_results(label_ids, preds, class_names, original_path, output_path):
        filename = os.path.join(original_path, "test.json")
        with open(filename) as infile:
            test_data = json.load(infile)

        if data_args.task_name in ["laptops", "restaurants", "restaurants16", "tweets"]:
            output_data = []
            data_id = 0
            text_index = 0
            aspect_index = 1
            label_index = 2

            for (original_data, true_label, pred_label) in zip(test_data[1:], label_ids, preds):
                text = original_data[text_index]
                aspect = original_data[aspect_index]
                label = int(original_data[label_index])
                assert label == true_label
                single_data = {"id": data_id, "text": text, "aspect words": aspect, "true label": class_names[true_label], "predicted label": class_names[pred_label]}
                output_data.append(single_data)
                data_id += 1
        
        elif data_args.task_name in ["agnews", "trec", "sst2", "subj", "dbpedia", "snli", "sst-5", "yahoo", 'mnli', 'mrpc', 'qqp']:
            output_data = []
            data_id = 0
            text_index = 0
            label_index = 1

            for (original_data, true_label, pred_label) in zip(test_data[1:], label_ids, preds):
                text = original_data[text_index]
                label = int(original_data[label_index])
                assert label == true_label
                single_data = {"id": data_id, "text": text, "true label": class_names[true_label], "predicted label": class_names[pred_label]}
                output_data.append(single_data)
                data_id += 1
        
        output_path = os.path.join(output_path, 'data_file')
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        output_filename = os.path.join(output_path, f"{data_args.test_mode}_pred_test.json") 
        with open(output_filename, 'w') as f:
            json.dump(output_data, f) 
        
        print(f"Predictions of test data have to writen to {output_filename}")

    #########################################################################

    #########################################################################
    #########################################################################
    for name, param in model.named_parameters():
        if 'bias' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    #########################################################################
    #########################################################################
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  ###################
        compute_metrics=build_compute_metrics_fn(data_args.task_name) # return {"acc": (preds == labels).mean()}
    )

    ###################################
    # /laptops/
    training_args.output_dir = os.path.join(training_args.output_dir, data_args.task_name)
    if not os.path.isdir(training_args.output_dir):
        os.mkdir(training_args.output_dir)
    
    # /seed_100/
    training_args.output_dir = os.path.join(training_args.output_dir, f"seed_{training_args.seed}")
    if not os.path.isdir(training_args.output_dir):
        os.mkdir(training_args.output_dir)

    # /16_shot/
    training_args.output_dir = os.path.join(training_args.output_dir, f"{data_args.num_k}_shot")
    if not os.path.isdir(training_args.output_dir):
        os.mkdir(training_args.output_dir)

    saved_model_path = os.path.join(training_args.output_dir, f'saved_model_{training_args.max_steps}_{training_args.per_device_train_batch_size}')
    ###################################
    
    # Training
    if training_args.do_train:
        trainer.train(model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None)
        # Use the early stop, so do not save the model in the end (unless specify save_at_last)
        if training_args.save_at_last:
            trainer.save_model(saved_model_path)
 
        if True: # trainer.is_world_master():
            tokenizer.save_pretrained(saved_model_path)
            torch.save(model_args, os.path.join(saved_model_path, "model_args.bin"))
            torch.save(data_args, os.path.join(saved_model_path, "data_args.bin"))
        
        # Reload the best checkpoint (for eval)
        model = model_fn.from_pretrained(saved_model_path)
        model = model.to(training_args.device)
        trainer.model = model
        if data_args.prompt:
            model.label_word_list = torch.tensor(train_dataset.label_word_list).long().cuda()
        if output_modes_mapping[data_args.task_name] == 'regression':
            # lower / upper bounds
            model.lb, model.ub = bound_mapping[data_args.task_name]
        model.model_args = model_args
        model.data_args = data_args
        model.tokenizer = tokenizer

    # Evaluation
    final_result = {
        'time': str(datetime.today()),
    }
    

    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Validate ***")

        eval_datasets = [eval_dataset]

        for eval_dataset in eval_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(eval_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=eval_dataset)
            eval_result = output.metrics 

            output_path = os.path.join(training_args.output_dir, 'loss_acc_file')
            if not os.path.isdir(output_path):
                os.mkdir(output_path)

            output_eval_file = os.path.join(
                output_path, f"{data_args.test_mode}_eval_results.txt"
            )

            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(eval_dataset.args.task_name))
                for key, value in eval_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
                    final_result[eval_dataset.args.task_name + '_dev_' + key] = value
                    
            eval_results.update(eval_result)
        
        # acc = confusion_matrix_analysis(eval_dataset.num_sample, output)
        # assert acc == eval_result['eval_acc']
        

    test_results = {}
    if training_args.do_predict:
        logging.info("*** Test ***")
        test_datasets = [test_dataset]

        for test_dataset in test_datasets:
            trainer.compute_metrics = build_compute_metrics_fn(test_dataset.args.task_name)
            output = trainer.evaluate(eval_dataset=test_dataset)
            test_result = output.metrics

            output_path = os.path.join(training_args.output_dir, 'loss_acc_file')
            if not os.path.isdir(output_path):
                os.mkdir(output_path)

            output_test_file = os.path.join(
                output_path, f"{data_args.test_mode}_test_results.txt"
            )
            if True: # trainer.is_world_master():
                with open(output_test_file, "w") as writer:
                    logger.info("***** Test results {} *****".format(test_dataset.args.task_name))
                    for key, value in test_result.items():
                        logger.info("  %s = %s", key, value)
                        writer.write("%s = %s\n" % (key, value))
                        final_result[test_dataset.args.task_name + '_test_' + key] = value

                if training_args.save_logit:
                    predictions = output.predictions
                    num_logits = predictions.shape[-1]
                    logits = predictions.reshape([test_dataset.num_sample, -1, num_logits]).mean(axis=0)

                    logit_path = os.path.join(training_args.output_dir, 'logits')
                    if not os.path.isdir(logit_path):
                        os.mkdir(logit_path)
                    np.save(os.path.join(logit_path, "{}_logit.npy".format(data_args.test_mode)), logits)

            test_results.update(test_result)

        acc = confusion_matrix_analysis(test_dataset.num_sample, output, data_args.write_output_file)
        """
        assert acc == test_result['eval_acc']
        """

    with FileLock('log.lock'):
        with open('log', 'a') as f:
            final_result.update(vars(model_args))
            final_result.update(vars(training_args))
            final_result.update(vars(data_args))
            if 'evaluation_strategy' in final_result:
                final_result.pop('evaluation_strategy')
            f.write(str(final_result) + '\n')
    
    # return eval_results
  
if __name__ == "__main__":
    main()
