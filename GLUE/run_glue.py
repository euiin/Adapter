#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
from datasets import load_dataset, load_metric

from safetensors import safe_open
from safetensors.torch import load_model, save_model, load_file, save_file

import transformers
from transformers import (
    AdamW, #new
    get_linear_schedule_with_warmup, #new
    AutoConfig, #new
    AutoModel, #new
    AutoModelForCausalLM, #new
    AutoModelForSequenceClassification,
    AutoTokenizer,
    LlamaTokenizer, #new
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from peft import (  # noqa: E402
    LoraConfig,
    VeraConfig,
    PeftConfig,
    #DoraConfig,
    #BottleneckConfig,
    PrefixTuningConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    #prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_watch: Optional[str] = None
    wandb_log_model: Optional[str] = None

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


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
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    apply_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply LoRA or not."},
    )
    apply_vera: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply LoRA or not."},
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA alpha"},
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={"help": "LoRA r"},
    )
    lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of LoRA parameters."},
    )
    vera_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of LoRA parameters."},
    )
    apply_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply adapter or not."},
    )
    adapter_path: Optional[str] = field(
        default=None,
        metadata={"help": "The file path of adapter parameters."},
    )
    adapter_type: Optional[str] = field(
        default='houlsby',
        metadata={"help": "houlsby or pfeiffer"},
    )
    adapter_size: Optional[int] = field(
        default=64,
        metadata={"help": "8, 16, 32, 64"},
    )
    apply_bitfit: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply bitfit or not."},
    )
    reg_loss_wgt: Optional[float] = field(
        default=0.0,
        metadata={"help": "Regularization Loss Weight"},
    )
    masking_prob: Optional[float] = field(
        default=0.0,
        metadata={"help": "Token Masking Probability"},
    )
    target_modules: List[str] = None
    modules_to_save: List[str] = None
    trans_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to apply adapter to trans model or not."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    learning_rate: float = field(default=5e-5) 
    learning_rate_head: float = field(default=1e-5) 
    eval_strategy: str = field(default="steps") 
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    report_to: str = field(
        default=None, metadata={"help": "Turn on W&B logging"},
    )
    run_name: str = field(
        default=None, metadata={"help": "Name of the W&B run"},
    )
    logging_steps: int = field(
        default=None, metadata={"help": "How often to log to W&B"},
    )
    save_steps: int = field(
        default=None, metadata={"help": "How often to save to W&B"},
    )
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})

def print_nameof_trainable_parameters(model):
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
            print('Trainable parameter :', name)

def extract_layer_number(key):
    parts = key.split('.')
    for part in parts:
        if part.isdigit():
            return int(part)
    return None

def resize_tensor(tensor, size):
    if tensor.size == size:
        return tensor
    elif tensor.size > size:
        return tensor[:size]
    else:
        return np.pad(tensor, (0, size - tensor.size), 'constant')

def duplicate_near_layer(tensors, large_tensors):
    dup_tensors = {}
    for key in tensors.keys():
        layer_number = extract_layer_number(key)
        if layer_number is not None:
            dup_key1 = key.replace(f'.{layer_number}.', f'.{layer_number * 2}.')
            dup_key2 = key.replace(f'.{layer_number}.', f'.{layer_number * 2 + 1}.')
            # Flatten the tensors to 1D arrays
            tensor1 = tensors[key].cpu()
            
            # Resize or pad the tensors to the same shape
            # max_size = max(tensor1.size(0), tensor2.size(0))
            max_size = 1024
            tensor1 = torch.from_numpy(resize_tensor(tensor1.numpy(), max_size))
            
            # Make duplicate tensor
            dup_tensors[dup_key1] = tensor1.to('cuda')
            dup_tensors[dup_key2] = tensor1.detach().to('cuda')
        # dealing no integer layer like classifier.dense.weight and classifier.dense.bias etc.
        else:
            dup_tensors[key] = large_tensors[key]
    return dup_tensors

def trans_lambda_d_layer(tensors, large_tensors):
    dup_tensors = {}
    for key in tensors.keys():
        layer_number = extract_layer_number(key)
        if layer_number is not None and 'vera_lambda_d' in key:
            dup_key1 = key.replace(f'.{layer_number}.', f'.{layer_number * 2}.')
            dup_key2 = key.replace(f'.{layer_number}.', f'.{layer_number * 2 + 1}.')
            # Flatten the tensors to 1D arrays
            tensor1 = tensors[key].cpu()
            
            # Resize or pad the tensors to the same shape
            # max_size = max(tensor1.size(0), tensor2.size(0))
            max_size = 1024
            tensor1 = torch.from_numpy(resize_tensor(tensor1.numpy(), max_size))
            
            # Make duplicate tensor
            dup_tensors[dup_key1] = tensor1.to('cuda')
            dup_tensors[dup_key2] = tensor1.detach().to('cuda')

    for large_key in large_tensors.keys():
        layer_number = extract_layer_number(large_key)
        if layer_number is not None and 'vera_lambda_d' in large_key:
            tensor_b = large_tensors[large_key].cpu()

            dup_tensors[large_key] = tensor_b.to('cuda')
        # dealing no integer layer like classifier.dense.weight and classifier.dense.bias etc.
        elif layer_number is None:
            dup_tensors[large_key] = large_tensors[large_key]
            print('key :', large_key)

    return dup_tensors

def trans_lambda_b_layer(tensors, large_tensors):
    dup_tensors = {}
    for key in tensors.keys():
        layer_number = extract_layer_number(key)
        if layer_number is not None and 'vera_lambda_b' in key:
            dup_key1 = key.replace(f'.{layer_number}.', f'.{layer_number * 2}.')
            dup_key2 = key.replace(f'.{layer_number}.', f'.{layer_number * 2 + 1}.')
            # Flatten the tensors to 1D arrays
            tensor1 = tensors[key].cpu()
            
            # Resize or pad the tensors to the same shape
            # max_size = max(tensor1.size(0), tensor2.size(0))
            max_size = 1024
            tensor1 = torch.from_numpy(resize_tensor(tensor1.numpy(), max_size))
            
            # Make duplicate tensor
            dup_tensors[dup_key1] = tensor1.to('cuda')
            dup_tensors[dup_key2] = tensor1.detach().to('cuda')

    for large_key in large_tensors.keys():
        layer_number = extract_layer_number(large_key)
        if layer_number is not None and 'vera_lambda_b' in large_key:
            tensor_b = large_tensors[large_key].cpu()

            dup_tensors[large_key] = tensor_b.to('cuda')
        # dealing no integer layer like classifier.dense.weight and classifier.dense.bias etc.
        elif layer_number is None:
            dup_tensors[large_key] = large_tensors[large_key]
            print('key :', large_key)

    return dup_tensors

def clone_layer(large_tensors):
    dup_tensors = {}
    for key in large_tensors.keys():
        tensor = large_tensors[key].cpu()

        dup_tensors[key] = tensor.to('cuda')
    return dup_tensors

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Check if parameter passed or if set within environ
    use_wandb = data_args.wandb_project is not None or (
            "WANDB_PROJECT" in os.environ and os.environ["WANDB_PROJECT"] is not None
    )
    # Only overwrite environ if wandb param passed
    if use_wandb:
        if len(data_args.wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = data_args.wandb_project
        if len(data_args.wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = data_args.wandb_watch
        if len(data_args.wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = data_args.wandb_log_model

    # torch.use_deterministic_algorithms(training_args.use_deterministic_algorithms)
    # logger.info("use_deterministic_algorithms: " + str(torch.are_deterministic_algorithms_enabled()))

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    print("training_args.seed :", training_args.seed)
    # training_args.seed is 42

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files)
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if model_args.apply_lora:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            #cls_dropout=training_args.cls_dropout,
            apply_lora=model_args.apply_lora,
            lora_alpha=model_args.lora_alpha,
            lora_r=model_args.lora_r,
            apply_adapter=model_args.apply_adapter,
            adapter_type=model_args.adapter_type,
            adapter_size=model_args.adapter_size,
            reg_loss_wgt=model_args.reg_loss_wgt,
            masking_prob=model_args.masking_prob,
        )
    if model_args.apply_vera:
        print("VeRA init") 
        config = VeraConfig(
            r=model_args.lora_r,
            #lora_alpha=lora_alpha,
            target_modules=["query", "value"],
            save_projection=False,
            #vera_dropout=lora_dropout,
            bias="none",
            #task_type="CAUSAL_LM",
            modules_to_save=model_args.modules_to_save, 
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        num_labels=num_labels, # 이거 lora 있을 때도 double로 안 걸리는지는 check 해야함.
    )
    model.config.use_cache = False

    new_path = model_args.vera_path
    # if trans_adapter is True, then save the model with the trans adapter.
    if model_args.trans_adapter and training_args.do_eval:
        tensors = {}
        with safe_open(model_args.vera_path + "/adapter_model.safetensors", framework="pt", device=0) as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k) # loads the full tensor given a key

        tensors_large = {}
        with safe_open("checkpoint/roberta-large_rte_vera/model/adapter_model.safetensors", framework="pt", device=0) as f:
            for k in f.keys():
                tensors_large[k] = f.get_tensor(k) # loads the full tensor given a key
        # dup_tensors = duplicate_near_layer(tensors, tensors_large)
        dup_tensors = trans_lambda_b_layer(tensors, tensors_large)
        # dup_tensors = clone_layer(tensors_large)
        print('dup_tensors :', dup_tensors)
        new_path = model_args.vera_path.replace('model', 'trans_vera_b_model')
        new_model = new_path + "/adapter_model.safetensors"
        save_file(dup_tensors, new_model)

    # In here, recall the VeraModel. In VeraModel, there is veralayer.
    if model_args.apply_vera:
        if model_args.vera_path is not None and training_args.do_eval:
            # model.load_adapter(model_args.vera_path)
            config = PeftConfig.from_pretrained(new_path)
            # unmatch size error is here!
            model = PeftModel.from_pretrained(model, new_path)
            model.print_trainable_parameters()
            model = model.model # Unwrapping the PeftModel 
            # so that evaluation metric pass the RobertaForSequenceClassification directly.
        else:
            model = get_peft_model(model, config)
            del(model.base_model.vera_A)
            del(model.base_model.vera_B)
            model.print_trainable_parameters()
        model.to('cuda')
        print('model structure : ', model)
        
        print_nameof_trainable_parameters(model)

    # Initialize the optimizer with different learning rates for specific layers
    if model_args.apply_vera and training_args.do_train:
        optimizer = torch.optim.AdamW([
            {'params': model.base_model.base_model.parameters()},
            {'params': model.base_model.classifier.modules_to_save.parameters(), 'lr': training_args.learning_rate_head}
            ], lr=training_args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),lr=training_args.learning_rate)

    # Initialize the learning rate scheduler
    num_training_steps = len(datasets["train"]) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=num_training_steps
    )

    trainable_params = []
    # if model_args.apply_vera:
    #     trainable_params.append('lora')

    if model_args.apply_lora:
        if model_args.lora_path is not None:
            lora_state_dict = torch.load(model_args.lora_path)
            logger.info(f"Apply LoRA state dict from {model_args.lora_path}.")
            logger.info(lora_state_dict.keys())
            model.load_state_dict(lora_state_dict, strict=False)
        trainable_params.append('lora')

    if model_args.apply_adapter:
        if model_args.adapter_path is not None:
            adapter_state_dict = torch.load(os.path.join(model_args.adapter_path, 'pytorch_adapter.bin'))
            head_state_dict = torch.load(os.path.join(model_args.adapter_path, 'pytorch_model_head.bin'))
            added_state_dict = {}
            for k, v in adapter_state_dict.items():
                new_k = k.replace(data_args.task_name + '.', '').replace('adapter_down.0.', 'adapter_A.').replace('adapter_up.', 'adapter_B.').replace('.adapters.', '.adapter.')
                added_state_dict[new_k] = v
            for k, v in head_state_dict.items():
                new_k = k.replace('heads.' + data_args.task_name + '.1', 'classifier.dense').replace('heads.' + data_args.task_name + '.4', 'classifier.out_proj')
                added_state_dict[new_k] = v
            logger.info(f"Apply adapter state dict from {model_args.adapter_path}.")
            logger.info(added_state_dict.keys())
            missing_keys, unexpected_keys = model.load_state_dict(added_state_dict, strict=False)
            for missing_key in missing_keys:
                assert 'adapter' not in missing_key, missing_key + ' is missed in the model'
            assert len(unexpected_keys) == 0, 'Unexpected keys ' + str(unexpected_keys)
        trainable_params.append('adapter')

    if model_args.apply_bitfit:
        trainable_params.append('bias')

    if len(trainable_params) > 0:
        for name, param in model.named_parameters():
            if name.startswith('deberta') or name.startswith('roberta'):
                param.requires_grad = False
                for trainable_param in trainable_params:
                    if trainable_param in name:
                        param.requires_grad = True
                        break
            else:
                param.requires_grad = True

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    if model_args.apply_vera:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            optimizers=(optimizer, lr_scheduler),
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            optimizers=optimizer,
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            # Check the config from that potential checkpoint has the right number of labels before using it as a
            # checkpoint.
            if AutoConfig.from_pretrained(model_args.model_name_or_path).num_labels == num_labels:
                checkpoint = model_args.model_name_or_path

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(datasets["validation_mismatched"])

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

            output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()