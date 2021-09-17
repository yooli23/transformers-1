#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on summarization.
"""
# You can also adapt this script on your own summarization task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import random

import datasets
import nltk
import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from typing import List

import transformers
from accelerate import Accelerator
from filelock import FileLock
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the testing data."
    )
    
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text " "(useful for T5 models).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )

    parser.add_argument(
        "--max_target_length",
        type=int,
        default=128,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--val_max_target_length",
        type=int,
        default=None,
        help="The maximum total sequence length for validation "
        "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
        "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
        "param of ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    # parser.add_argument(
    #     "--max_length",
    #     type=int,
    #     default=128,
    #     help=(
    #         "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
    #         " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
    #     ),
    # )
    
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of returned sequences to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    # parser.add_argument(
    #     "--pad_to_max_length",
    #     action="store_true",
    #     help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    # )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_valid_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--max_test_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )


    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--max_length", type=int, default=512, help="max length"
    )

    parser.add_argument(
        "--pad_to_max_length", action="store_true", help="do pading"
    )

    parser.add_argument(
        "--ignore_pad_token_for_loss", type=bool, default=True, help="do pading"
    )

    parser.add_argument(
        "--logging_steps", type=int, default=500, help="do pading"
    )

    parser.add_argument(
        "--save_steps", type=int, default=5000, help="do pading"
    )

    parser.add_argument(
        "--save_every_checkpoint", action="store_true"
    )

    parser.add_argument(
        "--no_sentence_dist", default=False, action="store_true", help="not using sentence level dist score in evaluation from KGSF paper"
    )
    

    parser.add_argument(
        "--max_grad_norm", type=float, default=1.0, help="max_grad_norm"
    )

    parser.add_argument(
        "--no_kb", action="store_true"
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        help="Description to the experiment",
        default='exp',
    )


    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    
    
    if accelerator.is_local_main_process:
        config = dict(
        dataset_id = "openkg",
        infra = "aml",
        )
        key = 'e2b2fc2e39cbbe4d98183a2c05e485a52812c70c'
        os.environ['WANDB_API_KEY'] = key
        import wandb
        wandb.init(
        project="redial1",
        notes="pretraining",
        tags=["pretrain"],
        config=config,
        entity= 'yooli23')

        wandb.run.name = args.exp_name




    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config)

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = args.source_prefix if args.source_prefix is not None else ""
    max_length = args.max_length
    padding = "max_length" if args.pad_to_max_length else False
    surfix = ' =>'
    max_target_length = args.max_target_length

    def preprocess_function(examples):
        contextes = examples['context']
        responses = examples['response']
        topics = examples['mentioned_in_context']
        movie_labels = examples['labels']
        kbs = examples['entities']
        inputs = []
        for context, response, topic, movie_label, kb in zip(contextes, responses, topics, movie_labels, kbs):
            if args.no_kb:
                inputs.append(prefix + ' ' + context + ' => ')
            else:
                inputs.append(prefix + ' ' + context + ' ' + kb + ' => ')
        model_inputs = tokenizer(inputs, max_length=args.max_length, padding=padding, truncation=True)

        # labels = model_inputs
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(responses, max_length=max_target_length, padding=padding, truncation=True)
            movie_labels = tokenizer(movie_labels, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and args.ignore_pad_token_for_loss:
            labels["labels"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["movie_labels"] = movie_labels["input_ids"]
        return model_inputs

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    
    # del raw_datasets['train']
    # del raw_datasets['test']
    column_names = ['context','response','mentioned_in_context','labels','entities']
    lm_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=False,
        desc=f"Processing dataset",
    )

    train_dataset = lm_datasets["train"]
    # train_dataset = lm_datasets["validation"]
    eval_dataset = lm_datasets["validation"]
    test_dataset = lm_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    def postprocess_text(preds, labels):
        preds = [pred.strip().replace('Agent :','') for pred in preds]
        labels = [label.strip().replace('Agent :','') for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, test_dataloader
    )

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metric

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    global_steps = 0
    tr_loss, logging_loss = 0.0, 0.0
    for epoch in range(args.num_train_epochs):
        model.train()
        # for step, batch in enumerate(train_dataloader):
        for step, batch in enumerate(train_dataloader):
            global_steps += 1
            input_batch = dict(batch)
            del input_batch["movie_labels"]
            outputs = model(**input_batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            tr_loss += loss.item()
            accelerator.backward(loss)
            
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

            if step % args.logging_steps == 0:
                logger.info(tr_loss)
                logger.info(logging_loss)
                logger.info(f"  EVALERR:  {(tr_loss - logging_loss)/float(args.logging_steps)}")
                if accelerator.is_local_main_process:
                    wandb.log({'loss': tr_loss - logging_loss})
                logging_loss = tr_loss
                progress_bar.update(args.logging_steps)

            if args.output_dir is not None and global_steps % args.save_steps == 0 and global_steps > 0:
                print('hit store')
                accelerator.wait_for_everyone()
                if accelerator.is_local_main_process:               
                    checkpoint_prefix = 'checkpoint'
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_steps))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
        logger.info("Model evaluating...")
        model.eval()
        if args.val_max_target_length is None:
            args.val_max_target_length = args.max_target_length

        gen_kwargs = {
            "max_length": args.val_max_target_length if args.val_max_target_length is not None else config.max_length,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_return_sequences,
        }

        metric = load_metric("./rouge_metric.py")
        metric_bleu = load_metric("./bleu_metric.py")
        metric_dist = load_metric("./dist_metric.py")
        metric_recall = load_metric("./recall_metric.py")
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                movie_labels = batch["movie_labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)
                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()
                movie_labels = accelerator.gather(movie_labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_movie_labels = tokenizer.batch_decode(movie_labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                metric.add_batch(predictions=decoded_preds, references=decoded_labels)

                _decoded_preds = [i.split() for i in decoded_preds]
                _decoded_labels = [[i.split()] for i in decoded_labels]
                metric_bleu.add_batch(predictions=_decoded_preds, references=_decoded_labels)
                metric_dist.add_batch(predictions=_decoded_preds, references=_decoded_labels)
                
                _movie_labels = [i.split('|') for i in decoded_movie_labels]
                _decoded_preds_recall = [[i.split()] for i in decoded_preds]
                metric_recall.add_batch(predictions=_decoded_preds_recall, references=_movie_labels)

                if args.max_valid_steps and step >= args.max_valid_steps:
                    break
        result = metric.compute(use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        result = {k: round(v, 4) for k, v in result.items()}

        logger.info(result)

        result_bleu = metric_bleu.compute()
        logger.info(result_bleu)

        if args.no_sentence_dist:
            result_dist = metric_dist.compute(sentence_dist=False)
        else:
            result_dist = metric_dist.compute()
        logger.info(result_dist)

        result_recall = metric_recall.compute()
        logger.info(result_recall)

        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            wandb.log({'valid_bleu': result_bleu['bleu']})
            wandb.log({'valid_rouge': result['rougeL']})
            wandb.log({'valid_dist2': result_dist['dis2']})
            wandb.log({'valid_dist3': result_dist['dis3']})
            wandb.log({'valid_dist4': result_dist['dis4']})
            wandb.log({'valid_recall_1': result_recall['recall_1']})
            wandb.log({'valid_recall_10': result_recall['recall_10']})
            wandb.log({'valid_recall_50': result_recall['recall_50']})


        metric = load_metric("./rouge_metric.py")
        metric_bleu = load_metric("./bleu_metric.py")
        metric_dist = load_metric("./dist_metric.py")
        metric_recall = load_metric("./recall_metric.py")
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                generated_tokens = accelerator.unwrap_model(model).generate(
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )
                
                generated_tokens = accelerator.pad_across_processes(
                    generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                )
                labels = batch["labels"]
                movie_labels = batch["movie_labels"]
                if not args.pad_to_max_length:
                    # If we did not pad to max length, we need to pad the labels too
                    labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()
                movie_labels = accelerator.gather(movie_labels).cpu().numpy()

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                if isinstance(generated_tokens, tuple):
                    generated_tokens = generated_tokens[0]
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                decoded_movie_labels = tokenizer.batch_decode(movie_labels, skip_special_tokens=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
                metric.add_batch(predictions=decoded_preds, references=decoded_labels)
                # todo
                _decoded_preds = [i.split() for i in decoded_preds]
                _decoded_labels = [[i.split()] for i in decoded_labels]
                
                metric_bleu.add_batch(predictions=_decoded_preds, references=_decoded_labels)
                metric_dist.add_batch(predictions=_decoded_preds, references=_decoded_labels)
                
                _movie_labels = [i.split('|') for i in decoded_movie_labels]
                _decoded_preds_recall = [[i.split()] for i in decoded_preds]
                metric_recall.add_batch(predictions=_decoded_preds_recall, references=_movie_labels)

                if args.max_test_steps and step >= args.max_test_steps:
                    break
        result = metric.compute(use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        result = {k: round(v, 4) for k, v in result.items()}

        logger.info(result)

        result_bleu = metric_bleu.compute()
        logger.info(result_bleu)
        
        if args.no_sentence_dist:
            result_dist = metric_dist.compute(sentence_dist=False)
        else:
            result_dist = metric_dist.compute()
        logger.info(result_dist)

        result_recall = metric_recall.compute()
        logger.info(result_recall)

        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            wandb.log({'test_bleu': result_bleu['bleu']})
            wandb.log({'test_rouge': result['rougeL']})
            wandb.log({'test_dist2': result_dist['dis2']})
            wandb.log({'test_dist3': result_dist['dis3']})
            wandb.log({'test_dist4': result_dist['dis4']})
            wandb.log({'test_recall_1': result_recall['recall_1']})
            wandb.log({'test_recall_10': result_recall['recall_10']})
            wandb.log({'test_recall_50': result_recall['recall_50']})
        
        if args.output_dir is not None and args.save_every_checkpoint:
            accelerator.wait_for_everyone()
            # unwrapped_model = accelerator.unwrap_model(model)
            # unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
            # accelerator.wait_for_everyone()
            if accelerator.is_local_main_process:               
                checkpoint_prefix = 'checkpoint'
                output_dir = os.path.join(args.output_dir, '{}-epoch-{}'.format(checkpoint_prefix, epoch))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

                tokenizer.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s", output_dir)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()
