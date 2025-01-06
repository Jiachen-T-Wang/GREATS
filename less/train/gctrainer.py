import os
import sys
import math
import shutil
import torch
import time
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import TrainOutput, set_seed, has_length
# from transformers.file_utils import is_torch_tpu_available
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.utils import is_sagemaker_mp_enabled, is_datasets_available
import datasets


from transformers.trainer_callback import TrainerState, TrainerCallback
from transformers.trainer_pt_utils import get_model_param_count
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

# from ..train.helper import *
import numpy as np

# from transformers.integrations import is_fairscale_available
import logging
import json
import warnings
import time


from ..train.utils_ghost_dot_prod import compute_GradProd_GC_per_iter, greedy_selection, find_GClayers

# Configure logging at the root level of logging
logging.basicConfig(level=logging.INFO)  # You can adjust the logging level as needed

# Create a logger object for your module
logger = logging.getLogger(__name__)




class GCTrainer(Trainer):
    def __init__(self, test_dataset, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.test_dataset = test_dataset

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")

        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        num_train_epochs = 50

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)


        ### Layers that are trainable and will be GCed.
        trainable_layers = find_GClayers(model)

        ### Load Embedding or Reference Model for SBERT and RHOLoss
        if args.method == "SBERT":
            # Load a pre-trained model
            from sentence_transformers import SentenceTransformer
            emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        elif args.method == "RHOLoss":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            # Load Llama3.1 8B model as reference model
            ref_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            ref_tokenizer = AutoTokenizer.from_pretrained(ref_model_name)
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name)
            ref_model.to(args.device)


        total_batched_samples = 0

        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True


            step = -1
            for step, inputs in enumerate(epoch_iterator):

                total_batched_samples += 1

                model.train()


                ##### ***************************************** #####
                ##### Gradient Selection #####
                eval_dataloader = self.get_gc_eval_dataloader(self.eval_dataset, val_batchsize=2, shuffle=True)
                
                if args.method == 'GREATS':

                    start_time = time.time()
                    
                    # Get TracIN scores and reuse the forward/backward pass
                    tracin_local_score, similarity_local_score = compute_GradProd_GC_per_iter(
                            model, 
                            device=args.device, 
                            batch_train=inputs, 
                            validation_loader=eval_dataloader, 
                            optimizer=self.optimizer, 
                            trainable_layers=trainable_layers)
                    
                    print('Total Extra Time for GREATS: ', time.time()-start_time)

                    start_time = time.time()

                    lr = self.optimizer.param_groups[0]["lr"]
                    lr_to_be_use_1, lr_to_be_use_2 = lr, lr**2

                    selected_ind = greedy_selection(tracin_local_score*lr_to_be_use_1, 
                                                    similarity_local_score*lr_to_be_use_2, 
                                                    int(len(tracin_local_score)/args.fracinv))

                    inputs['input_ids'] = inputs['input_ids'][selected_ind]
                    inputs['attention_mask'] = inputs['attention_mask'][selected_ind]
                    inputs['labels'] = inputs['labels'][selected_ind]

                    print('Total Extra Time for GradSelection: ', time.time()-start_time)


                ##### ***************************************** #####
                elif args.method == "GradNorm":

                    tracin_local_score, similarity_local_score = compute_TracIN_GC_per_iter(
                            model, device=args.device, batch_data=inputs, validation_loader=eval_dataloader, optimizer=self.optimizer, 
                            trainable_layers=trainable_layers)
                    
                    tracin_local_score = np.diag(similarity_local_score)

                    selected_ind = greedy_selection(tracin_local_score, 
                                                    similarity_local_score*0, 
                                                    int(len(tracin_local_score)/2))

                    inputs['input_ids'] = inputs['input_ids'][selected_ind]
                    inputs['attention_mask'] = inputs['attention_mask'][selected_ind]
                    inputs['labels'] = inputs['labels'][selected_ind]
                    
                elif args.method == "MaxLoss":

                    with torch.no_grad():
                        losses = []
                        for i in range(self._train_batch_size):
                            shift_ids = inputs['input_ids'][[i]]
                            shift_labels = inputs['labels'][[i]]
                            outputs = model(shift_ids, labels=shift_labels)
                            loss = outputs.loss
                            losses.append(loss.item())

                    selected_ind = greedy_selection(np.array(losses), 
                                                    np.zeros((len(losses), len(losses))), 
                                                    int(len(losses)/2))
                    
                    inputs['input_ids'] = inputs['input_ids'][selected_ind]
                    inputs['attention_mask'] = inputs['attention_mask'][selected_ind]
                    inputs['labels'] = inputs['labels'][selected_ind]


                ##### ***************************************** #####
                elif args.method == "SBERT":

                    X_str = []
                    for j, indices in enumerate( inputs['input_ids'] ):
                        # print('Example {}'.format(j))
                        output = self.tokenizer.decode(indices, add_special_tokens=False)
                        # print('')
                        # print('******** Train Example starts ********')
                        # print(output)
                        # print('******** Train Example ends ********')
                        X_str.append(output)

                    X_val_str = []
                    for _, val_inputs in enumerate(eval_dataloader):
                        for j, indices in enumerate( val_inputs['input_ids'] ):
                            # print('Example {}'.format(j))
                            output = self.tokenizer.decode(indices, add_special_tokens=False)
                            # print('')
                            # print('******** Val Example starts ********')
                            # print(output)
                            # print('******** Val Example ends ********')
                            X_val_str.append(output)

                    embedding_train = emb_model.encode(X_str)
                    embedding_val = emb_model.encode(X_val_str)

                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity_score = cosine_similarity(embedding_train, embedding_val)
                    similarity_score = np.mean(similarity_score, axis=1)

                    selected_ind = greedy_selection(similarity_score, 
                                                    np.zeros((len(similarity_score), len(similarity_score))),
                                                    int(len(similarity_score)/2))

                    inputs['input_ids'] = inputs['input_ids'][selected_ind]
                    inputs['attention_mask'] = inputs['attention_mask'][selected_ind]
                    inputs['labels'] = inputs['labels'][selected_ind]


                ##### ***************************************** #####
                elif args.method == "RHOLoss":

                    with torch.no_grad():
                        losses = []
                        for i in range(self._train_batch_size):
                            shift_ids = inputs['input_ids'][[i]]
                            shift_labels = inputs['labels'][[i]]
                            outputs = model(shift_ids, labels=shift_labels)
                            loss = outputs.loss
                            losses.append(loss.item())

                    X_str = []
                    for j, indices in enumerate( inputs['input_ids'] ):
                        output = self.tokenizer.decode(indices, add_special_tokens=False)
                        X_str.append(output)

                    ref_losses = []
                    for text in X_str:

                        # Tokenize the input and target text
                        input_tokens = ref_tokenizer(text, return_tensors="pt")

                        # Ensure the input and target tokens are the same length
                        input_ids = input_tokens["input_ids"]

                        # Shift the target ids to the right by one
                        shift_ids = input_ids[:, :-1].contiguous()
                        shift_labels = input_ids[:, 1:].contiguous()

                        shift_ids = shift_ids.to(args.device)
                        shift_labels = shift_labels.to(args.device)

                        with torch.no_grad():
                            outputs = ref_model(shift_ids, labels=shift_labels)
                            loss = outputs.loss
                            ref_losses.append(loss.item())

                    rho_losses = np.array(losses) - np.array(ref_losses)

                    selected_ind = greedy_selection(rho_losses, 
                                                    np.zeros((len(rho_losses), len(rho_losses))),
                                                    int(len(rho_losses)/2))

                    inputs['input_ids'] = inputs['input_ids'][selected_ind]
                    inputs['attention_mask'] = inputs['attention_mask'][selected_ind]
                    inputs['labels'] = inputs['labels'][selected_ind]




                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)


                ##### ***************************************** #####
                ##### Training Step #####
                # For GREATS, while we can save the fwd and bwd pass by using the one from ghost,
                # it does not seem to be working with the current implementation of the gradient accumulation.
                # So here we just do another fwd and bwd pass over the selected data points for clean implementation.
                if True:
                    with self.accelerator.accumulate(model):
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    selected_loss.backward()
                    tr_loss_step = selected_loss.detach()

                print('tr_loss_step', tr_loss_step)


                if (args.logging_nan_inf_filter
                    and not False
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping
                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break

                ##### ***************************************** #####
                ##### Add Evaluation #####
                # Save the results every 5 steps
                if total_batched_samples % 64 == 0:
                            
                    #### Evaluate on validation and test data
                    model.eval()

                    losses = []
                    for step, batch in enumerate(eval_dataloader):
                        with torch.no_grad():
                            outputs = model(**batch)
                        loss = outputs.loss
                        losses.append(loss.item())

                    try:
                        # filtering out the nan values in the losses
                        losses = [loss for loss in losses if not math.isnan(loss)]
                        eval_loss = np.mean(losses)
                        eval_perplexity = math.exp(eval_loss)
                    except OverflowError:
                        eval_perplexity = float("inf")

                    print('')
                    logger.info(f" total steps {total_batched_samples}: eval_perplexity: {eval_perplexity} eval_loss: {eval_loss}")

                    test_dataloader = self.get_gc_eval_dataloader(self.test_dataset, val_batchsize=10)

                    losses = []
                    for step, batch in enumerate(test_dataloader):
                        with torch.no_grad():
                            outputs = model(**batch)
                        loss = outputs.loss
                        losses.append(loss.item())
                        if step == 10:
                            break

                    try:
                        losses = [loss for loss in losses if not math.isnan(loss)]
                        test_loss = np.mean(losses)
                        test_perplexity = math.exp(test_loss)
                    except OverflowError:
                        test_perplexity = float("inf")

                    logger.info(f" total steps {total_batched_samples}: test_perplexity: {test_perplexity} test_loss: {test_loss}")

                    #### For MMLU dataset, the choice of multiple answers are ["A", "B", "C", "D"]
                    acc = None
                    if self.args.analysis_dataset == "mmlu":
                        choices = ["A", "B", "C", "D"]
                        answer_choice_ids = [self.tokenizer.encode(" " + answer_choice, add_special_tokens=False)[-1] for answer_choice in choices]

                        from less.train.mmlu_eval import compute_accuracy
                        cors, acc, all_probs = compute_accuracy(args, model, self.tokenizer, answer_choice_ids=answer_choice_ids)

                        logger.info(f" total steps {total_batched_samples}: test_acc: {acc}")
                    
                    elif self.args.analysis_dataset == "tydiqa" and total_batched_samples % 100 == 0 and "Mistral" in args.result_dir and total_batched_samples > 1900:
                        from less.train.tydiqa_eval import compute_accuracy
                        acc = compute_accuracy(args, model, self.tokenizer)
                        logger.info(f" total steps {total_batched_samples}: test_acc: {acc}")

                    #### Save Results
                    file_path = args.result_dir

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
                        "train_loss": tr_loss.item() / len(train_dataloader) if isinstance(tr_loss, torch.Tensor) else tr_loss / len(train_dataloader),
                        "test_accuracy": acc,
                        "epoch": epoch,
                        "step": total_batched_samples
                    }

                    # Append the new data entry to the list
                    data.append(new_entry)

                    # Write the updated data back to the file
                    with open(file_path, "w") as file:
                        json.dump(data, file, indent=4)
                        
                ##### ***************************************** #####


            # if step < 0:
            #     logger.warning(
            #         "There seems to be not a single sample in your epoch_iterator, stopping training at step"
            #         f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
            #         f" num_steps ({max_steps}) higher than the number of available samples."
            #     )
            #     self.control.should_training_stop = True

            # self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            # self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            # if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            #     if False:
            #         # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            #         xm.master_print(met.metrics_report())
            #     else:
            #         logger.warning(
            #             "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
            #             "configured. Check your training configuration if this is unexpected."
            #         )
            # if self.control.should_training_stop:
            #     break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if False:
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)


    def get_gc_eval_dataloader(self, eval_dataset=None, val_batchsize=1, shuffle=False) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": val_batchsize,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            # dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, shuffle=shuffle, **dataloader_params))


    def print_example(self, indices):

        output = self.tokenizer.decode(indices, add_special_tokens=False)
        print('')
        print('******** Example starts ********')
        print(output)
        print('******** Example ends ********')