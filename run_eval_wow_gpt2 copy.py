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

import logging
import math
import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, list_metrics, load_metric

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import re
import pandas as pd


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.10.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
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

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    test_file: Optional[str] = field(default=None, metadata={"help": "The input test data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    kg_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional knowledge sequence length"
        },
    )
    add_triples_in_checked_sentence: bool = field(
        default=False,
        metadata={
            "help": "Whether to add triples as knowledge from checked sentence in the input or not."
        },
    )
    add_triples_in_text: bool = field(
        default=False,
        metadata={
            "help": "Whether to add triples as knowledge from context in the input or not."
        },
    )
    res_file: str = field(
        default=None,
        metadata={"help": "saving result file name"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None and self.test_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
    extension = (
        data_args.test_file.split(".")[-1]
    )
    if extension == "txt":
        extension = "text"
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = config.eos_token_id

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    column_names = raw_datasets["test"].column_names
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    print("debugging")
    print(tokenizer.encode("Wizard:", return_tensors="pt"))
    print(torch.cat((tokenizer.encode("Apprentice:", return_tensors="pt"), tokenizer.encode("I love you", return_tensors="pt"), torch.tensor([[config.eos_token_id]])), 1))

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = {}
            list_ids = []
            list_reference = []
            for idx, example in enumerate(examples['text']):
                ids = []
                if data_args.add_triples_in_text:
                    text_triples = []
                for turn_idx, turn in enumerate(example):
                    speaker = turn['speaker'].split('_')[-1]
                    if not (ids) and speaker =="Wizard":
                        ids = tokenizer.encode("Apprentice:") + tokenizer.encode(examples['topic'][idx]) + [config.eos_token_id]
                    if speaker == 'Wizard':
                        if data_args.add_triples_in_checked_sentence:
                            # encode triples in checked sentence and append it to ids
                            triples = "None "
                            if 'triples_in_checked_sentence' in turn:
                                for triples_idx, triple in enumerate(turn['triples_in_checked_sentence']):
                                    if triples_idx < 20:
                                        if triples == "None ":
                                            triples = (triple['subject'] + " " + triple['relation'] + " " + triple['object'] + " ")
                                        else:
                                            triples += (triple['subject'] + " " + triple['relation'] + " " + triple['object']+ " ")
                            else:
                                raise ValueError("--add_triples_in_checked_sentence requires a file including triples_in_checked_sentence key")
                            info_ids = tokenizer.encode("Triples:") + tokenizer.encode(triples)
                            if len(info_ids) > data_args.kg_length:
                                info_ids = info_ids[:data_args.kg_length]
                            ids += info_ids
                        if data_args.add_triples_in_text:
                            # encode triples in context and append it to ids
                            triples = "None "
                            for triples_idx, triple in enumerate(text_triples):
                                if triples_idx < 20:
                                    if triples == "None ":
                                        triples = (triple['subject'] + " " + triple['relation'] + " " + triple['object'] + " ")
                                    else:
                                        triples += (triple['subject'] + " " + triple['relation'] + " " + triple['object'] + " ")
                            info_ids = tokenizer.encode("Triples:") + tokenizer.encode(triples)
                            if len(info_ids) > data_args.kg_length:
                                info_ids = info_ids[:data_args.kg_length]
                            ids += info_ids
                        if (ids):
                            list_ids.append(ids)
                        list_reference.append(turn['turn_text'])
                        ids = ids + tokenizer.encode("Wizard:") + tokenizer.encode(turn['turn_text']) + [config.eos_token_id]
                    else:
                        ids = ids + tokenizer.encode("Apprentice:") + tokenizer.encode(turn['turn_text']) + [config.eos_token_id]
                    if data_args.add_triples_in_text:
                        # add triples in context
                        if 'triples_in_text' in turn:
                            # add triples in current turn ahead of previous triples
                            text_triples = turn['triples_in_text'] + text_triples
                        else:
                            raise ValueError("--add_triples_in_text requires a file including triples_in_text key")
            output["input_ids"] = list_ids
            output["reference"] = list_reference
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    print("debugging")
    print(len(tokenized_datasets["test"]))
    print(tokenized_datasets["test"][:3])
    for ids in tokenized_datasets["test"][:3]["input_ids"]:
        print(tokenizer.decode(ids))
    print(tokenized_datasets["test"][:3]["reference"])


    test_dataset = tokenized_datasets["test"]
    if data_args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(data_args.max_test_samples))

    model.eval()
    metrics = load_metric('bleu')
    list_candidate = []
    list_reference = []

    for idx, row in enumerate(test_dataset):
        input_ids = row["input_ids"]
        prompt_text = tokenizer.decode(input_ids)
        input_ids += tokenizer.encode("Wizard:")
        input_ids = torch.tensor([input_ids]).to(torch.int64)
        input_ids = input_ids.to(model.device)
        # prompt_text = row["text"]
        print(prompt_text)
        reference = row["reference"]
        generated_text_samples = model.generate(
            input_ids=input_ids, 
            max_length=block_size,  
            num_return_sequences=1,
            repetition_penalty=1.0,
            top_p=0.9,
            temperature=1.0,
            do_sample=True,
            top_k=0,
        )
        # print(generated_text_samples)
        candidate = ""
        for i, beam in enumerate(generated_text_samples):
            print(f"=== GENERATED SEQUENCE ===")
            beam = beam.tolist()
            text = tokenizer.decode(beam, clean_up_tokenization_spaces=True)
            candidate = text.replace(prompt_text, "").replace("Wizard:", "").replace("<|endoftext|>", "")
            print(candidate)
            
            # metrics.add(prediction = candidate, reference = [reference])
        list_candidate.append(candidate)
        list_reference.append(reference)
    # Create DataFrame
    df = pd.DataFrame({'candidate': list_candidate, 'reference': list_reference})
    df.to_csv(data_args.res_file)

    # print(metrics.compute(max_order=4))


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
