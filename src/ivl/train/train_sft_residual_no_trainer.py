# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import json
import logging
import math
import os
from dataclasses import dataclass, field
import json
import torch
import transformers
from typing import Optional

import datasets
import torch
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
)

logger = get_logger(__name__)

from torch.nn import CrossEntropyLoss
from data_module import *

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def calculate_loss(logits, pretrained_logits, labels) -> torch.Tensor:

    vocab_size = logits.shape[-1]

    logits = pretrained_logits[:,:,:vocab_size] + logits
    logits = logits.float()

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    return loss

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    pretrained_model_name_or_path: Optional[str] = field(default=None)
    use_flash_attn: bool = False
    add_special_tokens: Optional[str] = None


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    template: str = field(default="vicuna")
    new_sys_message: str = field(
        default=None
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

def main():

    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps
    )

    # Make one log on every process with the configuration for debugging.

    logging.basicConfig(
        format="[%(asctime)s,%(msecs)03d] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        logger.info(f"model_args={model_args}")
        logger.info(f"data_args={data_args}")
        logger.info(f"training_args={training_args}")

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # Handle the repository creation
    if accelerator.local_main_process_first:
        os.makedirs(training_args.output_dir, exist_ok=True)

        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
        )

        if config.model_type == "gemma2": # assume same architecture for both models
            attn_implementation = "eager"
        else:
            attn_implementation = "flash_attention_2"
        

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            use_cache=False,
        )

        pretrained_model = AutoModelForCausalLM.from_pretrained(
            model_args.pretrained_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
            use_cache=False,
        )

        pretrained_model = pretrained_model.eval()

        if training_args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True
        )

        if tokenizer.pad_token != tokenizer.unk_token:
            tokenizer.pad_token = tokenizer.unk_token

        if data_args.template.startswith("qwen"):
            tokenizer.pad_token = "<|endoftext|>"
            tokenizer.unk_token = "<unk>"
            tokenizer.eos_token = "<|im_end|>"

        data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        data_module["train_dataset"], 
        shuffle=True, 
        collate_fn=default_data_collator, 
        batch_size=training_args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        data_module["eval_dataset"], 
        collate_fn=default_data_collator, 
        batch_size=training_args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=training_args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)


    if training_args.max_steps is None or training_args.max_steps == -1:
        training_args.max_steps = int(training_args.num_train_epochs) * num_update_steps_per_epoch
        overrode_max_steps = True

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps * accelerator.num_processes,
        num_training_steps=training_args.max_steps
        if overrode_max_steps
        else training_args.max_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    pretrained_model = accelerator.prepare(pretrained_model)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)

    if overrode_max_steps:
        training_args.max_steps = int(training_args.num_train_epochs) * num_update_steps_per_epoch

    # Train!
    total_batch_size = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(data_module['train_dataset'])}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(training_args.max_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    state = []
    
    for epoch in range(int(training_args.num_train_epochs)):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"]
                }

                labels = batch["labels"]

                outputs = model(**inputs)
                pretrained_outputs = pretrained_model(**inputs)

                logits = outputs.logits
                pretrained_logits = pretrained_outputs.logits

                loss = calculate_loss(logits, pretrained_logits, labels)

                log_dict = {"loss": round(loss.item(),2), "epoch": epoch, "step": step, "learning_rate": lr_scheduler.get_last_lr()[0]}
                logger.info(log_dict)
                state.append(log_dict)
                accelerator.log(log_dict, step=step)

                total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather_for_metrics(loss.repeat(training_args.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        try:
            eval_loss = torch.mean(losses)
            perplexity = math.exp(eval_loss)
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        accelerator.log(
            {
                "perplexity": perplexity,
                "eval_loss": eval_loss,
                "train_loss": total_loss.item() / len(train_dataloader),
                "epoch": epoch,
                "step": completed_steps,
            },
            step=completed_steps,
        )

        if training_args.save_strategy == "epoch":
            accelerator.wait_for_everyone()
            output_dir = f"epoch_{epoch}"
            if training_args.output_dir is not None:
                output_dir = os.path.join(training_args.output_dir, output_dir)
                os.makedirs(output_dir, exist_ok=True)
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.bin"))
            torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = accelerator.get_state_dict(model)
            if accelerator.is_main_process:
                unwrapped_model.save_pretrained(
                    output_dir, state_dict=state_dict,
                    is_main_process=accelerator.is_main_process, safe_serialization=True
                )
                del unwrapped_model
                del state_dict
                tokenizer.save_pretrained(output_dir)
                trainer_state = {
                    "completed_steps": completed_steps,
                    "epoch": epoch,
                    "state": state,
                }
                with open(os.path.join(output_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
                    json.dump(trainer_state, f, indent=4)

    accelerator.end_training()

    if training_args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(
                training_args.output_dir, state_dict=state_dict,
                is_main_process=accelerator.is_main_process, safe_serialization=True
            )
            tokenizer.save_pretrained(training_args.output_dir)
            trainer_state = {
                "completed_steps": completed_steps,
                "epoch": epoch,
                "state": state,
            }
            with open(os.path.join(training_args.output_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
                json.dump(trainer_state, f)

if __name__ == "__main__":
    main()
