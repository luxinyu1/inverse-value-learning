# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from dataclasses import dataclass, field
import pathlib
from typing import Dict, Optional, Sequence
import transformers
from transformers import Trainer

from modeling_llama import LlamaCascade

from data_module import *

def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=False, rank0_only=False)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    pretrained_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
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
    debug_mode: bool = False


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

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_flash_attn:
        model = LlamaCascade.from_pretrained(
            model_name_or_path=model_args.model_name_or_path,
            pretrained_model_name_or_path=model_args.pretrained_model_name_or_path,
            attn_implementation="flash_attention_2"
        )
    else:
        model = LlamaCascade.from_pretrained(
            model_name_or_path=model_args.model_name_or_path,
            pretrained_model_name_or_path=model_args.pretrained_model_name_or_path,
        )

    local_rank = training_args.local_rank

    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
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
    elif data_args.template.startswith("llama-3"):
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.add_special_tokens is not None:
        special_tokens_dict  = {"additional_special_tokens": model_args.add_special_tokens.split(" ")}
        rank0_print("Adding special tokens: {}".format(special_tokens_dict))
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)

if __name__ == "__main__":
    train()