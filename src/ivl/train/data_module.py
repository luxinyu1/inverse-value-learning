import torch
import transformers
from conversation import SeparatorStyle, get_conv_template
from transformers.trainer_pt_utils import LabelSmoother
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
import json
from typing import Optional
import numpy as np
from multiprocessing import Pool

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

LLAMA_2_CHAT_DEFAULT_SYSTEM_MESSAGE = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def apply_prompt_template(sources, template, sys_messages, new_sys_message):
    conv = get_conv_template(template)

    if conv.sep_style == SeparatorStyle.LLAMA2:
        conv.system_message = LLAMA_2_CHAT_DEFAULT_SYSTEM_MESSAGE # set the empty system_message in fschat==0.28.0

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    dynamic_system_message = False
    if new_sys_message is not None and sys_messages is None:
        rank0_print(f"Replace original system message with '{new_sys_message}'")
        conv.system_message = new_sys_message
    elif sys_messages is not None and new_sys_message is None:
        assert len(sources) == len(sys_messages)
        dynamic_system_message = True
        rank0_print("Dynamic system message training ...")
    elif sys_messages is not None and new_sys_message is not None:
        raise ValueError()

    # Apply prompt templates
    rank0_print(sources[0]) # [[{"from": "human", "value":}, {"from": "gpt", "value":}, ...]]
    conversations = []
    if dynamic_system_message:
        for i, (source, sys) in enumerate(zip(sources, sys_messages)): # instance loop
            conv.system_message = sys
            if roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source): # dialog loop
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
    else:
        for i, source in enumerate(sources): # instance loop
            if source[0]["from"] == "system" or roles[source[0]["from"]] != conv.roles[0]:
                # Skip the first one if it is not from human
                source = source[1:]

            conv.messages = []
            for j, sentence in enumerate(source): # dialog loop
                role = roles[sentence["from"]]
                assert role == conv.roles[j % 2], f"{i}"
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())
    return conversations, conv

def tokenize_conversations(conversations, tokenizer):
    rank0_print("Rendered Sample[0]:", conversations[0])

    rank0_print("Max length:", tokenizer.model_max_length)

    # Tokenize all conversations
    input_ids = tokenizer(
        conversations,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids # [dataset_len, max_seq_len]

    targets = input_ids.clone() # [dataset_len, max_seq_len]

    rank0_print("Tokenized Sample[0]:", input_ids[0].tolist())

    return input_ids, targets


def mask_targets(conversations, targets, tokenizer, conv):
    # Mask targets

    first = True

    if conv.sep_style == SeparatorStyle.LLAMA2:
        sep = conv.roles[1] + " " # [/INST]
        for conversation, target in zip(conversations, targets):
            
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(conv.sep2) # " </s><s>"

            cur_len = 1
            target[:cur_len] = IGNORE_TOKEN_ID

            for i, turn in enumerate(turns):
                if turn == "":
                    break
                parts = turn.split(sep)

                if len(parts) != 2:
                    break

                parts[0] += sep
                round_len = len(tokenizer(turn).input_ids) + 2
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                if False:
                    rank0_print(tokenizer.convert_ids_to_tokens(target[cur_len : cur_len + instruction_len]))

                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

                cur_len += round_len

            target[cur_len:] = IGNORE_TOKEN_ID

            if first:
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                rank0_print(tokenizer.decode(z))
                rank0_print(tokenizer.convert_ids_to_tokens(z))
                first = False

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    rank0_print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    

    elif conv.sep_style == SeparatorStyle.CHATML and isinstance(tokenizer, transformers.models.qwen2.tokenization_qwen2.Qwen2Tokenizer):
        sep = conv.roles[1] + "\n"
        # print("sep:", sep)
        for conversation, target in zip(conversations, targets):
            
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split("<|im_start|>user\n")

            round_sep_len = len(tokenizer("<|im_start|>user\n").input_ids) - 1 # <s>

            system_len = len(tokenizer(turns[0]).input_ids)

            # first mask the system part
            cur_len = system_len
            target[:cur_len] = IGNORE_TOKEN_ID

            turns = turns[1:]

            for i, turn in enumerate(turns):

                if turn == "":
                    rank0_print("Skip one.")
                    break
                parts = turn.split(sep)

                if len(parts) != 2:
                    rank0_print("Skip one.")
                    break

                parts[0] += sep

                round_len = len(tokenizer(turn).input_ids) + round_sep_len + 1
                instruction_len = len(tokenizer(parts[0]).input_ids) + round_sep_len + 1

                if False:
                    z = target.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                    rank0_print("1:", tokenizer.convert_ids_to_tokens(tokenizer("\n" + turn).input_ids))
                    rank0_print("2:", tokenizer.convert_ids_to_tokens(tokenizer("\n" + parts[0]).input_ids))
                    rank0_print("3:", tokenizer.convert_ids_to_tokens(z))
                    rank0_print("4:", tokenizer.convert_ids_to_tokens(z[cur_len : cur_len + instruction_len]))
                    rank0_print("5:", tokenizer.convert_ids_to_tokens(z[cur_len : cur_len + round_len]))

                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

                cur_len += round_len

            target[cur_len:] = IGNORE_TOKEN_ID

            if first:
                z = target.clone()
                z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
                rank0_print(tokenizer.decode(z))
                rank0_print(tokenizer.convert_ids_to_tokens(z))
                first = False

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    rank0_print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    elif conv.sep_style == SeparatorStyle.LLAMA3:

        sep = f"<|start_header_id|>{conv.roles[1]}<|end_header_id|>\n\n"

        for conversation, target in zip(conversations, targets):
            
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            turns = conversation.split(f"<|start_header_id|>{conv.roles[0]}<|end_header_id|>\n\n")

            round_sep_len = len(tokenizer(f"<|start_header_id|>{conv.roles[0]}<|end_header_id|>\n\n").input_ids) - 1 # <s>

            system_len = len(tokenizer(turns[0]).input_ids)

            # first mask the system part
            cur_len = system_len
            target[:cur_len] = IGNORE_TOKEN_ID

            turns = turns[1:]

            for i, turn in enumerate(turns):

                if turn == "":
                    rank0_print("Skip one.")
                    break

                parts = turn.split(sep)
                if len(parts) != 2:
                    rank0_print("Skip one.")
                    break

                parts[0] += sep

                round_len = len(tokenizer(turn).input_ids) + round_sep_len - 1
                instruction_len = len(tokenizer(parts[0]).input_ids) + round_sep_len - 1

                if first:
                    z = target.clone()
                    z = torch.where(z == IGNORE_TOKEN_ID, 128001, z)
                    rank0_print("1:", tokenizer.convert_ids_to_tokens(z))
                    rank0_print("2:", tokenizer.convert_ids_to_tokens(z[cur_len : cur_len + instruction_len]))
                    rank0_print("3:", tokenizer.convert_ids_to_tokens(z[cur_len : cur_len + round_len]))
                    first = False

                target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

                cur_len += round_len

            target[cur_len:] = IGNORE_TOKEN_ID

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_TOKEN_ID
                    rank0_print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    return targets

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    template: str,
    new_sys_message=None,
    sys_messages=None
) -> Dict:

    if len(sources) <= 1000:
        conversations, conv = apply_prompt_template(sources, template, sys_messages, new_sys_message)
        input_ids, targets = tokenize_conversations(conversations, tokenizer)
        targets = mask_targets(conversations, targets, tokenizer, conv)
    else:  # If the data volume is large, use multithreading for processing
        with Pool(processes = 8) as p:
            conversations, conv = p.apply_async(
                apply_prompt_template, (sources, template, sys_messages, new_sys_message)
            ).get()
            input_ids, targets = p.apply_async(
                tokenize_conversations, (conversations, tokenizer)
            ).get()
            targets = p.apply_async(
                mask_targets, (conversations, targets, tokenizer, conv)
            ).get()
            p.close()
            p.join()
            
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                raw_data, 
                tokenizer: transformers.PreTrainedTokenizer,
                template: str,
                new_sys_message=None):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        if "system" in raw_data[0]: 
            sys_messages = [example["system"] for example in raw_data]
            data_dict = preprocess(sources, tokenizer, template, sys_messages=sys_messages)
        else:
            data_dict = preprocess(sources, tokenizer, template, new_sys_message=new_sys_message)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                raw_data,
                tokenizer: transformers.PreTrainedTokenizer,
                template: str,
                new_sys_message=None):
        
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}
        self.template = template
        self.new_sys_message = new_sys_message

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        if "system" in self.raw_data[i]: 
            ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.template, sys_messages=[raw_data[i]["system"]])
        else:
            ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer, self.template, new_sys_message=self.new_sys_message)

        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    raw_data = json.load(open(data_args.data_path, "r"))

    if data_args.debug_mode:
        raw_data = raw_data[:1000]

    # Split train/test
    np.random.seed(0)
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * 0.98)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    if data_args.new_sys_message is not None:
        rank0_print(f"Setting all sys message with: '{data_args.new_sys_message}'")

    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer, template=data_args.template, new_sys_message=data_args.new_sys_message)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer, template=data_args.template, new_sys_message=data_args.new_sys_message)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)