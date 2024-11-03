# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys

from functools import partial
from typing import Any, Dict, Optional, Tuple
from warnings import warn

import torch
from omegaconf import DictConfig

from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, utils
from torchtune.modules.peft.peft_utils import (
    get_adapter_params,
    get_merged_lora_ckpt,
    set_trainable_params,
    validate_missing_and_unexpected_for_lora,
)
from torchtune.modules.peft.lora import LoRALinear
from torchtune.recipe_interfaces import FTRecipeInterface
from tqdm import tqdm
import math
import os
import json
import time
from torchtune import config, utils
import evaluate
metric = evaluate.load("rouge")


# from torchtune.utils_ghost_dot_prod import *
# from torchtune.layers.lora_layers import GCLoRALinear

from utils_ghost_dot_prod import *
from layers.lora_layers import GCLoRALinear

log = utils.get_logger("DEBUG")


class LoRAFinetuneRecipeSingleDevice(FTRecipeInterface):
    """
    LoRA finetuning recipe for dense transformer-based LLMs such as Llama2. This recipe is optimized
    for single GPU training. Training on CPU is not supported.

    Features:
        - Activation Checkpointing. This can be controlled using the ``activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.g

        - Gradient Accumulation. You can simulate larger batch sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

                Total Batch Size = batch_size * gradient accumulation steps.

            For example: with batch_size=1 and gradient_accumulation_steps=32 we get a total batch size of 32.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Lower precision optimizers. This recipe supports lower-precision optimizers from the bitsandbytes
            library (https://huggingface.co/docs/bitsandbytes/main/en/index). We've tested the recipe with
            8-bit AdamW and Paged AdamW.

        - Checkpointing. Model weights are checkpointed both at the end of each epoch and at the end of
            training. Currently we checkpoint both the adapter weights (trainable params only) and the
            complete merged weights (adapter weights added back to the base model). For more details
            please take a look at our LoRA tutorial
            (https://pytorch.org/torchtune/main/tutorials/lora_finetune.html).

            Optimizer State and recipe state (seed, total_epochs, number of epochs run etc) are
            only saved at the end of a given epoch and used in case of resuming training. Resuming
            training is controlled by the ``resume_from_checkpoint`` flag. Mid-epoch checkpointing is
            currently not supported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/tutorials/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

    For a full list of example configs for this recipe, run ``tune ls`` on the command line. Each config
    has example commands for how to kick-off training.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        ValueError: If ``dtype`` is set to fp16.
        RuntimeError: If ``dtype`` is set to bf16 and the hardware does not support bf16.

    """

    def __init__(self, cfg: DictConfig) -> None:

        self._device = utils.get_device(device=cfg.device)
        # Reduced precision logic
        self._dtype = utils.get_dtype(cfg.dtype, device=self._device)
        # fp16 precision is explicitly disabled as it is not supported in this
        # recipe (for example, no gradient scaling).
        if self._dtype == torch.float16:
            raise ValueError(
                "fp16 precision is not supported in this recipe. Please use fp32 or bf16."
            )
        # For CUDA devices, check if the HW supports bf16 if bf16 is specified.
        if (
            self._dtype == torch.bfloat16
            and self._device != torch.device("cpu")
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("Full bf16 training is not supported on this hardware.")
        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.log_every_n_steps if cfg.log_every_n_steps else 1
        self._log_peak_memory_every_n_steps = 100

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = utils.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.total_training_steps = 0

        self.save_cpt = cfg.save_cpt
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps

        self.method = cfg.method
        self.fracinv = cfg.fracinv
        self.batch_size = cfg.batch_size
        self.n_val = cfg.n_val
        self.n_test = cfg.n_test

        if self.method == 'TracIN-AdaptiveSelect-PerBatch':
            self.batch_size = int( self.batch_size * self.fracinv )

        self.result_dir = cfg.result_dir


        self.result_dir = self.result_dir + '{}-BS{}-Nval{}'.format(
            self.method, self.batch_size, self.n_val)
        
        # Add LoRA parameters
        self.result_dir = self.result_dir + '-LR{}-LoRA_R{}_Alpha{}'.format(
            cfg.optimizer.lr, cfg.model.lora_rank, cfg.model.lora_alpha)
        
        if self.method == 'TracIN-AdaptiveSelect-PerBatch':
            self.result_dir = self.result_dir + '-FRACINV{}'.format(self.fracinv)

        if self.save_cpt is True:
            self.result_dir = self.result_dir + '_cpt_results.json'
        else:
            self.result_dir = self.result_dir + '_results.json'

        if os.path.exists( self.result_dir ):
            os.remove( self.result_dir )
            print(f"The file {self.result_dir} has been removed.")


    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. This includes the
        base model weights. If resume_from_checkpoint is True, this also includes
        the adapter weights and recipe state
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            if utils.ADAPTER_KEY not in checkpoint_dict:
                raise ValueError(
                    "Adapter weights not found. Please ensure a valid adapter checkpoint is provided."
                )
            # _update_recipe_state will throw an exception if the recipe state is not corrctly loaded
            # no need to check here
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        # If seed, total_epoch or max_steps_per_epoch don't match,
        # warn the user and overwrite
        if (
            self.seed != ckpt_dict[utils.SEED_KEY]
            or self.total_epochs != ckpt_dict[utils.TOTAL_EPOCHS_KEY]
            or self.max_steps_per_epoch != ckpt_dict[utils.MAX_STEPS_KEY]
        ):
            warn(
                message="""Configured value for seed, epochs or max_steps_per_epoch
                does not match the value stored in checkpoint."""
            )
        self.seed = utils.set_seed(seed=ckpt_dict[utils.SEED_KEY])
        self.epochs_run = ckpt_dict[utils.EPOCHS_KEY]
        self.total_epochs = ckpt_dict[utils.TOTAL_EPOCHS_KEY]
        self.max_steps_per_epoch = ckpt_dict[utils.MAX_STEPS_KEY]

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe state. This includes recipe state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, learning rate scheduler, sampler, and dataloader.
        """
        self._metric_logger = config.instantiate(cfg.metric_logger)

        # log config with parameter override
        self._metric_logger.log_config(cfg)

        self._model_compile = cfg.compile
        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            compile_model=cfg.compile,
            base_model_state_dict=checkpoint_dict[utils.MODEL_KEY],
            lora_weights_state_dict=(
                checkpoint_dict[utils.ADAPTER_KEY]
                if self._resume_from_checkpoint
                else None
            ),
        )



        ##### ***************************************** #####
        ##### Change to make the model work with TracIN #####

        print(self._model)
        self.replace_LoRALinear(self._model)
        print(self._model)
        
        ##### ***************************************** #####

        self._tokenizer = config.instantiate(cfg.tokenizer)
        log.info("Tokenizer is initialized from file.")

        self._optimizer = self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=(
                checkpoint_dict[utils.OPT_KEY] if self._resume_from_checkpoint else None
            ),
        )
        log.info("Optimizer is initialized.")

        self._loss_fn = config.instantiate(cfg.loss)
        log.info("Loss is initialized.")


        ##### ***************************************** #####
        ##### Make train, validation and test set #####

        # Dataloader depends on the tokenizer and loss_fn and should be
        # setup after all of these are setup
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset=cfg.dataset,
            shuffle=cfg.shuffle,
            batch_size=self.batch_size,
        )
        log.info("Training Dataset and Sampler are initialized.", len(self._dataloader))

        self.val_sampler, self.val_dataloader = self._setup_data(
            cfg_dataset=cfg.val_dataset,
            shuffle=True,
            batch_size=cfg.val_batch_size,
            validation=True,
        )
        log.info("Validation Dataset and Sampler are initialized.", len(self.val_dataloader))

        self.test_sampler, self.test_dataloader = self._setup_data(
            cfg_dataset=cfg.test_dataset,
            shuffle=False,
            batch_size=cfg.test_batch_size,
        )
        log.info("Test Dataset and Sampler are initialized.", len(self.test_dataloader))
        
        ##### ***************************************** #####


        # Finally update the recipe state which can only be correctly set after all of the
        # other components have been initialized and updated.

        # Number of training steps in each epoch depends on the number of batches produced
        # by the dataloader and the max_steps_per_epoch param set by the user and is used
        # for logging and tracking training state. This should be computed after the dataloader
        # has been setup
        self._steps_per_epoch = (
            len(self._dataloader) // self._gradient_accumulation_steps
        )
        if (
            self.max_steps_per_epoch is not None
            and self.max_steps_per_epoch < self._steps_per_epoch
        ):
            self._steps_per_epoch = self.max_steps_per_epoch
            self.total_training_steps = self.epochs_run * self._steps_per_epoch

        # Learning rate scheduler can only be set up after number of steps has been computed
        self._lr_scheduler = self._setup_lr_scheduler(
            cfg_lr_scheduler=cfg.lr_scheduler,
            num_training_steps=self.total_epochs * self._steps_per_epoch,
            last_epoch=self.total_training_steps - 1,
        )

        self._profiler_enabled = cfg.profiler.enabled
        self._profiler = config.instantiate(cfg.profiler)

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        compile_model: bool,
        base_model_state_dict: Dict[str, Any],
        lora_weights_state_dict: Optional[Dict[str, Any]] = None,
    ) -> nn.Module:
        with utils.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg_model)

        self._lora_rank = cfg_model.lora_rank
        self._lora_alpha = cfg_model.lora_alpha
        self.lora_dropout = 0.1
        self.adapter_params = get_adapter_params(model)
        set_trainable_params(model, self.adapter_params)

        if enable_activation_checkpointing:
            utils.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerDecoderLayer}
            )

        base_missing, base_unexpected = model.load_state_dict(
            base_model_state_dict, strict=False
        )
        if lora_weights_state_dict:
            lora_missing, lora_unexpected = model.load_state_dict(
                lora_weights_state_dict, strict=False
            )
        else:
            lora_missing, lora_unexpected = None, None

        validate_missing_and_unexpected_for_lora(
            lora_attn_modules=cfg_model.lora_attn_modules,
            apply_lora_to_mlp=cfg_model.apply_lora_to_mlp,
            apply_lora_to_output=getattr(cfg_model, "apply_lora_to_output", False),
            base_missing=base_missing,
            base_unexpected=base_unexpected,
            lora_missing=lora_missing,
            lora_unexpected=lora_unexpected,
        )
        # Validate model adapter params were loaded in with the expected dtype
        # TODO (rohan-varma): Further validation to ensure the appropriate base params
        # are NF4 vs bf16 based on the quantization config.
        utils.validate_expected_param_dtype(
            self.adapter_params.items(), dtype=self._dtype
        )

        log.info(f"Model is initialized with precision {self._dtype}.")
        # Compile model, if enabled.
        if compile_model:
            log.info("Compiling model with torch.compile...")
            model = utils.wrap_compile(model)
        if self._device.type == "cuda":
            memory_stats = utils.get_memory_stats(device=self._device)
            utils.log_memory_stats(memory_stats)

        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            optimizer.load_state_dict(opt_state_dict)

        log.info("Optimizer and loss are initialized.")
        return optimizer

    def _setup_lr_scheduler(
        self,
        cfg_lr_scheduler: DictConfig,
        num_training_steps: int,
        last_epoch: int,
    ) -> Optimizer:
        lr_scheduler = config.instantiate(
            cfg_lr_scheduler,
            self._optimizer,
            num_training_steps=num_training_steps,
            last_epoch=last_epoch,
        )

        log.info("Learning rate scheduler is initialized.")
        return lr_scheduler

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        shuffle: bool,
        batch_size: int,
        validation=False,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports
        Map-style Datasets which fit into memory and an option for random shuffling.
        Samplers, iterable datasets, and streaming datasets are not supported.
        """
        ds = config.instantiate(
            cfg_dataset,
            tokenizer=self._tokenizer,
        )

        if validation:
            ds = ds.subset(start=0, end=self.n_val)
        print(cfg_dataset)
        if "grammar" in cfg_dataset["_component_"]:
            if not validation:
                ds = ds.subset(start=self.n_val, end=self.n_val+self.n_test)

        sampler = DistributedSampler(
            ds,
            num_replicas=1,
            rank=0,
            shuffle=shuffle,
            seed=0,
        )
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            collate_fn=partial(
                utils.padded_collate,
                padding_idx=self._tokenizer.pad_id,
                ignore_idx=self._loss_fn.ignore_index,
            ),
        )

        print('Load Data with length {}'.format(len(ds)))

        return sampler, dataloader


    def save_checkpoint(self, epoch: int) -> None:
        """
        Checkpoint the state of the recipe. The constructed checkpoint state dict
        contains the following information:
        - Merged weights with key MODEL_KEY
        - Adapter weights with key ADAPTER_KEY
        - Relevant recipe state if training is not complete

        Checkpointer will save the merged weights, adapter weights and recipe state in
        different checkpoint files. To correctly resume from training, the adapter weights
        and recipe state must be provided along with the base model weights.
        """
        ckpt_dict = {}
        # if training is in-progress, checkpoint the optimizer state as well
        if epoch + 1 < self.total_epochs:
            ckpt_dict.update(
                {
                    utils.OPT_KEY: self._optimizer.state_dict(),
                    utils.SEED_KEY: self.seed,
                    utils.EPOCHS_KEY: self.epochs_run,
                    utils.TOTAL_EPOCHS_KEY: self.total_epochs,
                    utils.MAX_STEPS_KEY: self.max_steps_per_epoch,
                }
            )

        # Move to CPU to avoid a copy on GPU
        state_dict = {k: v.cpu() for k, v in self._model.state_dict().items()}

        # Construct the full state dict with LoRA weights merged into base LLM weights
        merged_state_dict = get_merged_lora_ckpt(
            state_dict,
            rank=self._lora_rank,
            alpha=self._lora_alpha,
        )
        ckpt_dict.update({utils.MODEL_KEY: merged_state_dict})

        # Construct the adapter weights
        adapter_key_filter = lambda x: x in self.adapter_params
        adapter_state_dict = {
            k: v for k, v in self._model.state_dict().items() if adapter_key_filter(k)
        }
        ckpt_dict.update({utils.ADAPTER_KEY: adapter_state_dict})
        self._checkpointer.save_checkpoint(
            ckpt_dict,
            epoch=epoch,
            intermediate_checkpoint=(epoch + 1 < self.total_epochs),
        )

    def train(self) -> None:
        """
        The core training loop.
        """

        #### check the trainable layers ### 
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                print(name)            

        N_VAL_ITER = 5
        
        trainable_layers = find_GClayers(self._model)

        if self._model_compile:
            log.info("NOTE: torch.compile is enabled and model is compiled in first forward. Expect a relatively slow first iteration.")

        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):
            # Update the sampler to ensure data is correctly shuffled across epochs in case shuffle is True
            self._sampler.set_epoch(curr_epoch)

            # Optionally profile the training loop
            with self._profiler:
                for idx, batch in enumerate(pbar := tqdm(self._dataloader)):

                    self._model.train()

                    if (
                        self.max_steps_per_epoch is not None
                        and (idx // self._gradient_accumulation_steps)
                        == self.max_steps_per_epoch
                    ):
                        break

                    if self._profiler_enabled:
                        self._profiler.step()

                    input_ids, labels = batch
                    if self.total_training_steps % 3000 == 0:
                        self.get_rouge_score(input_ids.to(self._device), labels.to(self._device))

                    if self.method == "TracIN-AdaptiveSelect-PerBatch":

                        # Update the sampler to ensure data is correctly shuffled across epochs in case shuffle is True
                        self.val_sampler.set_epoch( curr_epoch*len(self._dataloader) + idx )

                        tracin_local_score, similarity_local_score = compute_TracIN_GC_per_iter(
                                self._model, device=self._device, batch_data=batch, validation_loader=self.val_dataloader, 
                                optimizer=self._optimizer, trainable_layers=trainable_layers, loss_fn = self._loss_fn)

                        lr = self._optimizer.param_groups[0]["lr"]
                        lr_to_be_use_1, lr_to_be_use_2 = lr, lr**2

                        selected_ind = greedy_selection(tracin_local_score*lr_to_be_use_1, 
                                                        similarity_local_score*lr_to_be_use_2, 
                                                        int(len(tracin_local_score)/self.fracinv))
                    
                        input_ids, labels = input_ids[selected_ind], labels[selected_ind]

                    elif self.method == "GradNorm":

                        # Update the sampler to ensure data is correctly shuffled across epochs in case shuffle is True
                        self.val_sampler.set_epoch( self.total_training_steps )

                        tracin_local_score, similarity_local_score = compute_TracIN_GC_per_iter(
                                self._model, device=self._device, batch_data=batch, validation_loader=self.val_dataloader, 
                                optimizer=self._optimizer, trainable_layers=trainable_layers, loss_fn = self._loss_fn)
                        
                        tracin_local_score = np.diag(similarity_local_score)

                        selected_ind = greedy_selection(tracin_local_score, 
                                                        similarity_local_score*0, 
                                                        int(len(tracin_local_score)/self.fracinv))

                        input_ids, labels = input_ids[selected_ind], labels[selected_ind]

                    elif self.method == "MaxLoss":

                        input_ids = input_ids.to(self._device)
                        labels = labels.to(self._device)

                        with torch.no_grad():
                            logits = self._model(input_ids)
                            # Shift so that tokens < n predict n
                            logits = logits[..., :-1, :].contiguous()
                            labels_temp = labels[..., 1:].contiguous()
                            logits = logits.transpose(1, 2)
                            # Compute loss
                            per_sample_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
                            losses = per_sample_loss_fn(logits, labels_temp) #### torch.Size([16, 128256, 311]) torch.Size([16, 311])

                        losses = losses.cpu().tolist()
                        losses = np.mean(losses, axis=1)

                        selected_ind = greedy_selection(losses, 
                                                        np.zeros((len(losses), len(losses))), 
                                                        int(len(losses)/self.fracinv))
                        
                        input_ids, labels = input_ids[selected_ind], labels[selected_ind]

                    input_ids = input_ids.to(self._device)
                    labels = labels.to(self._device)

                    logits = self._model(input_ids)
                    # Shift so that tokens < n predict n
                    logits = logits[..., :-1, :].contiguous()
                    labels = labels[..., 1:].contiguous()
                    logits = logits.transpose(1, 2)
                    # Compute loss
                    loss = self._loss_fn(logits, labels) #### torch.Size([16, 128256, 311]) torch.Size([16, 311])

                    tr_loss = loss.item()

                    print(f"Epoch {curr_epoch+1}| Iter {idx+1}| Training Loss: {loss.item()}")

                    loss = loss / self._gradient_accumulation_steps
                    loss.backward()

                    if (idx + 1) % self._gradient_accumulation_steps == 0:
                        self._optimizer.step()
                        self._optimizer.zero_grad(set_to_none=True)
                        self._lr_scheduler.step()
                        # Update the number of steps when the weights are updated
                        self.total_training_steps += 1

                    if self.save_cpt:
                        self.save_checkpoint(epoch=curr_epoch)
                        print('Checkpoint Saved')


                    # def logits_to_text(tokenizer, logits):
                    #     # Assuming logits are batched and you are using the highest probability indices
                    #     predicted_indices = logits.argmax(dim=-1)  # Get the predicted token indices
                    #     summaries = [tokenizer.decode(ids) for ids in predicted_indices]
                    #     return summaries

                    # def ids_to_text(tokenizer, labels):
                    #     # Convert label indices to text, assuming labels are batched
                    #     labels = [ids for ids in labels if ids != -100]
                    #     summaries = tokenizer.decode(labels)
                    #     return summaries

                    # def batch_ids_to_text(tokenizer, batchlabels):
                    #     # Convert label indices to text, assuming labels are batched
                    #     summaries = [ids_to_text(tokenizer, labels) for labels in batchlabels]
                    #     return summaries


                    ##### ***************************************** #####
                    ##### Add Evaluation #####
                    # Save the results every 5 steps
                    if self.total_training_steps % 50 == 0:
                                
                        #### Evaluate on validation and test data
                        self._model.eval()

                        losses = []
                        for i, val_batch in enumerate(self.val_dataloader):
                            input_ids, labels = val_batch
                            input_ids = input_ids.to(self._device)
                            labels = labels.to(self._device)

                            with torch.no_grad():
                                logits = self._model(input_ids)
                                logits = logits[..., :-1, :].contiguous()
                                labels = labels[..., 1:].contiguous()
                                logits = logits.transpose(1, 2)
                                loss = self._loss_fn(logits, labels) #### torch.Size([16, 128256, 311]) torch.Size([16, 311])
                            losses.append( loss.item() )

                        try:
                            eval_loss = np.mean(losses)
                            eval_perplexity = math.exp(eval_loss)
                        except OverflowError:
                            eval_perplexity = float("inf")

                        losses = []
                        for i, val_batch in enumerate(self.test_dataloader):
                            input_ids, labels = val_batch
                            input_ids = input_ids.to(self._device)
                            labels = labels.to(self._device)

                            with torch.no_grad():
                                logits = self._model(input_ids)
                                logits = logits[..., :-1, :].contiguous()
                                labels = labels[..., 1:].contiguous()
                                logits = logits.transpose(1, 2)
                                preds = torch.argmax(logits, dim=1)
                                # print(preds[10], labels[10]) 
                                loss = self._loss_fn(logits, labels) #### torch.Size([16, 128256, 311]) torch.Size([16, 311])
                            losses.append( loss.item() )

                            if i > 25:
                                with torch.no_grad():
                                    if self.total_training_steps % 300 == 0:
                                        self.get_rouge_score(input_ids, labels)
                                    break

                        try:
                            test_loss = np.mean(losses)
                            test_perplexity = math.exp(test_loss)
                        except OverflowError:
                            test_perplexity = float("inf")

                        print(f"Epoch {curr_epoch+1}| Iter {idx+1}| Val Perplexity: {eval_perplexity}| Test Perplexity: {test_perplexity}")


                        #### Save Results
                        file_path = self.result_dir

                        # Read the existing data, if available
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            with open(file_path, "r") as file:
                                data = json.load(file)
                        else:
                            data = []

                        # Create new data entry
                        new_entry = {
                            "test_perplexity": test_perplexity.item() if isinstance(test_perplexity, torch.Tensor) else test_perplexity,
                            "test_loss": test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss,
                            "eval_perplexity": eval_perplexity.item() if isinstance(eval_perplexity, torch.Tensor) else eval_perplexity,
                            "eval_loss": eval_loss.item() if isinstance(eval_loss, torch.Tensor) else eval_loss,
                            "train_loss": tr_loss.item() if isinstance(tr_loss, torch.Tensor) else tr_loss,
                            "epoch": curr_epoch+1,
                            "step": self.total_training_steps
                        }

                        # Append the new data entry to the list
                        data.append(new_entry)

                        # Write the updated data back to the file
                        with open(file_path, "w") as file:
                            json.dump(data, file, indent=4)
                            
                    ##### ***************************************** #####


                    # # Log peak memory for iteration
                    # if (self.total_training_steps % self._log_peak_memory_every_n_steps == 0 and self._device.type == "cuda"):
                    #     # Log peak memory for iteration
                    #     memory_stats = utils.get_memory_stats(device=self._device)
                    #     self._metric_logger.log_dict(
                    #         memory_stats, step=self.total_training_steps
                    #     )

            self.epochs_run += 1
            if self.save_cpt:
                self.save_checkpoint(epoch=curr_epoch)
                print('Checkpoint Saved')



    def cleanup(self) -> None:
        self._metric_logger.close()

    def replace_LoRALinear(self, module):
        for layer_str in dir(module):
            layer = getattr(module, layer_str)
            if type(layer) == LoRALinear:
                new_layer = GCLoRALinear(in_features=layer.in_dim, 
                                        out_features=layer.out_dim, 
                                        r=self._lora_rank, 
                                        lora_alpha=self._lora_alpha, 
                                        lora_dropout=self.lora_dropout, 
                                        device='cuda')

                new_layer.weight = layer.weight
                if layer.bias is not None:
                    new_layer.bias = layer.bias
                # set new_layer.weight to be trainable
                new_layer.weight.requires_grad = False
                del layer
                print('Found LoRA Layer: {}'.format(layer_str))
                setattr(module, layer_str, new_layer)
                print('Replaced LoRA Layer: {}'.format(layer_str))
        if hasattr(module,'children'):
            for immediate_child_module in module.children():
                self.replace_LoRALinear(immediate_child_module)

    def get_rouge_score(self, input_ids, labels):

        if "Llama-2" in self._output_dir:
            prompt_list, ground_truth_list = [], []
            for i in range(input_ids.size(0)):
                text = self._tokenizer.decode(input_ids[i].tolist())
                index = text.find('### Response:')
                prompt = text[:index]
                ground_truth = text[index+13:]         
                prompt_list.append(prompt)
                ground_truth_list.append(ground_truth)
            output_list = []
            for i in range(len(input_ids)):
                prompt_token = self._tokenizer.encode(prompt_list[i])
                prompt = input_ids[i][:len(prompt_token)]
                generated_tokens = utils.generate(
                    model=self._model,
                    prompt=prompt,
                    temperature=0.6,
                    eos_id=self._tokenizer.eos_id,
                    custom_generate_next_token=None,
                    max_generated_tokens=300
                )
                output = self._tokenizer.decode(generated_tokens)
                pp = self._tokenizer.decode(input_ids[i].tolist())
                pp2 = self._tokenizer.decode(prompt.tolist())
                output = output.replace(pp2, '')
                output_list.append(output)
        else:
            # get the generated text
            prompt_list, ground_truth_list = [], []
            for i in range(input_ids.size(0)):
                text = self._tokenizer.decode(input_ids[i].tolist())
                index = text.find('assistant<|end_header_id|>')
                prompt = text[:index+26]
                ground_truth = text[index+26:]            
                prompt_list.append(prompt)
                ground_truth_list.append(ground_truth)

            output_list = []
            for i in range(len(input_ids)):
                # print(f"Prompt: {input_ids[i]}")
                index_eos = (labels[i] == 128007).nonzero(as_tuple=True)[-1]
                prompt_len = index_eos[0] +2
                prompt = input_ids[i][:prompt_len] 
                print(f"Prompt: {prompt}")  
                # print(self._tokenizer.encode("\n",add_bos=False,add_eos=False), self._tokenizer.encode("\n\n",add_bos=False,add_eos=False), self._tokenizer.encode("\n\n\n",add_bos=False,add_eos=False))

                generated_tokens = utils.generate(
                model=self._model,
                prompt=prompt,
                temperature=0,
                eos_id=[128009],
                custom_generate_next_token=None,
                max_generated_tokens=300
                )
                # print(f"Generated Tokens: {generated_tokens}")
                output = self._tokenizer.decode(generated_tokens)
                print(f"Output: {output}")  
                

                pp = self._tokenizer.decode(input_ids[i].tolist())
                pp2 = self._tokenizer.decode(prompt.tolist())
                # print(f"Prompt: {pp}, \n\nPrompt2: {pp2}, \n\nOutput: {output}")
                output = output.replace(pp2, '')
                output_list.append(output)
                if i == 0:
                    print(f"Prompt: {pp2}, \n\nOutput: {output}") 

        print(output_list,ground_truth_list)
        # save the generated text
        with open('output.txt', 'w') as f:
            for item in output_list:
                f.write("%s\n" % item)
        with open('ground_truth.txt', 'w') as f:
            for item in ground_truth_list:
                f.write("%s\n" % item)
        scores = metric.compute(predictions=output_list, references=ground_truth_list)
        print(scores)



@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    config.log_config(recipe_name="LoRAFinetuneRecipeSingleDevice", cfg=cfg)
    recipe = LoRAFinetuneRecipeSingleDevice(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()



if __name__ == "__main__":
    sys.exit(recipe_main())
