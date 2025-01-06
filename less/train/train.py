#!/usr/bin/env python
# coding=utf-8
import logging
import os
import random
import sys
import time

import datasets
import torch
import torch.distributed as dist
import transformers
# from instruction_tuning.train.lora_trainer import LoRAFSDPTrainer, Trainer

import peft
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser, Trainer,
                          set_seed)


from less.data_selection.get_training_dataset import get_training_dataset
from less.train.data_arguments import DataArguments, get_data_statistics
from less.train.model_arguments import ModelArguments, add_padding_to_tokenizer
from less.train.training_arguments import TrainingArguments

logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from ..layers.linear import GCLinear
from ..layers.lora_layers import GCLoRALinear
from ..train.gctrainer import GCTrainer





def main():

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Dataset parameters {data_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Load training dataset
    train_dataset = get_training_dataset(data_args.train_files,
                                         tokenizer=tokenizer,
                                         max_seq_length=data_args.max_seq_length,
                                         sample_percentage=data_args.percentage,
                                         seed=data_args.sample_data_seed)
    print('Training Set')
    get_data_statistics(train_dataset)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, torch_dtype=model_args.torch_dtype)
    add_padding_to_tokenizer(tokenizer)

    # resize embeddings if needed (e.g. for LlamaTokenizer)
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        # if you load lora model and resize the token embeddings, the requires_grad flag is set to True for embeddings
        if isinstance(model, PeftModel):
            model.get_input_embeddings().weight.requires_grad = False
            model.get_output_embeddings().weight.requires_grad = False





    if not isinstance(model, PeftModel) and model_args.lora:

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=model_args.lora_target_modules,
        )

        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA to model.")
        model.print_trainable_parameters()

        # for checkpointing
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)




    ##### ***************************************** #####
    ##### Change to make the model work with "Ghost Inner-product" #####

    def replace_Linear(module, last_layer_only=False):
        for layer_str in dir(module):
            layer = getattr(module, layer_str)

            if type(layer) == torch.nn.Linear:
                new_layer = GCLinear(in_features=layer.in_features, out_features=layer.out_features)
                new_layer.weight = layer.weight
                new_layer.bias = layer.bias
                del layer
                setattr(module, layer_str, new_layer)

        if not last_layer_only:
            if hasattr(module,'children'):
                for immediate_child_module in module.children():
                    replace_Linear(immediate_child_module)

    def replace_LoRALinear(module):
        for layer_str in dir(module):
            layer = getattr(module, layer_str)
    
            if type(layer) == peft.tuners.lora.layer.Linear:
                new_layer = GCLoRALinear(in_features=layer.in_features, out_features=layer.out_features, 
                                         r=model_args.lora_r, 
                                         device='cuda')
                new_layer.weight = layer.weight
                del layer
                setattr(module, layer_str, new_layer)

        if hasattr(module,'children'):
            for immediate_child_module in module.children():
                replace_LoRALinear(immediate_child_module)

    if model_args.lora:
        replace_LoRALinear(model)
    else:
        replace_Linear(model, last_layer_only=True)

        ##### ***************************************** #####
        ##### Make the last layer trainable #####
        for param in model.base_model.model.lm_head.parameters():
            param.requires_grad = True
        ##### ***************************************** #####

    model.print_trainable_parameters()
    ##### ***************************************** #####




    if "dataset" in train_dataset.features:
        train_dataset = train_dataset.remove_columns(["dataset", "id", "messages"])            

    ##### ***************************************** #####
    ##### Make validation and test set #####
    ##### Choice of validation: ["bbh", "tydiqa", "mmlu"]
    from less.data_selection.get_validation_dataset import get_dataset
    data_args.data_dir = './data'

    analysis_dataset = get_dataset(training_args.analysis_dataset,
                                    data_dir=data_args.data_dir,
                                    tokenizer=tokenizer,
                                    max_length=data_args.max_seq_length, 
                                    validation=True, 
                                    k = training_args.n_val, 
                                    subject = training_args.subject)

    test_dataset = get_dataset(training_args.analysis_dataset,
                                data_dir=data_args.data_dir,
                                tokenizer=tokenizer,
                                max_length=data_args.max_seq_length, 
                                validation=False, 
                                k = training_args.n_test, 
                                subject = training_args.subject)
    ##### ***************************************** #####

    training_args.result_dir = '/scratch/gpfs/tw8948/GREATS_Result/'
    
    if training_args.method == 'GREATS':
        training_args.per_device_train_batch_size = int( training_args.per_device_train_batch_size * training_args.fracinv )

    if "Mistral-7B" in model_args.model_name_or_path:
        training_args.result_dir = training_args.result_dir + 'Mistral-7B'
    elif "Llama-2" in model_args.model_name_or_path:
        training_args.result_dir = training_args.result_dir + 'Llama-2'

    training_args.result_dir = training_args.result_dir + '-{}-BS{}-TrainPct{}-{}-NVAL{}-NTEST{}'.format(
        training_args.method, 
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps, 
        data_args.percentage, 
        training_args.subject, 
        training_args.n_val, 
        training_args.n_test)
    
    # Add LoRA parameters
    training_args.result_dir = training_args.result_dir + '-LoRA_R{}_Alpha{}_Dropout{}'.format(
        model_args.lora_r, model_args.lora_alpha, model_args.lora_dropout,
    )
    
    training_args.result_dir = training_args.result_dir + '-LR{}'.format(training_args.learning_rate)
    training_args.result_dir = training_args.result_dir + '-Seed{}'.format(training_args.seed)
    
    if training_args.method == 'GREATS':
        training_args.result_dir = training_args.result_dir + '-FRACINV{}'.format(training_args.fracinv)

    training_args.result_dir = training_args.result_dir + '_results.json'

    if os.path.exists( training_args.result_dir ):
        os.remove( training_args.result_dir )
        print(f"The file {training_args.result_dir} has been removed.")

    trainer = GCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=analysis_dataset,
        test_dataset=test_dataset, 
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")
    )

    # Training
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # remove the full model in the end to save space, only adapter is needed
    if isinstance(model, PeftModel):
        pytorch_model_path = os.path.join(
            training_args.output_dir, "pytorch_model_fsdp.bin")
        os.remove(pytorch_model_path) if os.path.exists(
            pytorch_model_path) else None

if __name__ == "__main__":
    main()
