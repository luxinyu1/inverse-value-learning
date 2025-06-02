import torch
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast, 
    PreTrainedTokenizerFast
)
import editdistance
from fastchat import conversation
import json
from argparse import ArgumentParser
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
from dataclasses import dataclass, field

parser = ArgumentParser()

parser.add_argument("--pretrained-tokenizer", type=str, required=True)
parser.add_argument("--value-tokenizer", type=str, required=True)
parser.add_argument("--token-alignment-matrix-file", type=str, default="./data/alignment_matrix_normalized.pt")
parser.add_argument("--pure-text-file", type=str, required=True)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--num-workers", type=int, default=32)
parser.add_argument("--first-n", type=int, default=None)

args = parser.parse_args()

def sigmoid(x):
    """Compute the sigmoid."""
    return 1.0 / (1 + np.exp(-x))

base_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer)
value_tokenizer = AutoTokenizer.from_pretrained(args.value_tokenizer)

base_vocab = base_tokenizer.get_vocab()
value_vocab = value_tokenizer.get_vocab()

base_tokenizer.pad_token = base_tokenizer.eos_token
value_tokenizer.pad_token = value_tokenizer.eos_token

TOKENIZER_TO_SPECIAL_TOKEN = {
    LlamaTokenizer: "▁", 
    LlamaTokenizerFast: "Ġ",
    PreTrainedTokenizerFast: "Ġ",
}

def dtw(series_1, series_2, norm_func=np.linalg.norm):
    """
    Use dynamic time wrapping to align to tokenizers, modified from:
    https://github.com/talcs/simpledtw/blob/master/simpledtw.py
    """
    matrix = np.zeros((len(series_1) + 1, len(series_2) + 1))
    matrix[0, :] = np.inf
    matrix[:, 0] = np.inf
    matrix[0, 0] = 0
    for i, vec1 in enumerate(series_1):
        for j, vec2 in enumerate(series_2):
            cost = norm_func(vec1, vec2)
            matrix[i + 1, j + 1] = cost + min(
                matrix[i, j + 1], matrix[i + 1, j], matrix[i, j]
            )
    matrix = matrix[1:, 1:]
    i = matrix.shape[0] - 1
    j = matrix.shape[1] - 1
    matches = []
    mappings_series_1 = [list() for v in range(matrix.shape[0])]
    mappings_series_2 = [list() for v in range(matrix.shape[1])]
    while i > 0 or j > 0:
        matches.append((i, j))
        mappings_series_1[i].append(j)
        mappings_series_2[j].append(i)
        option_diag = matrix[i - 1, j - 1] if i > 0 and j > 0 else np.inf
        option_up = matrix[i - 1, j] if i > 0 else np.inf
        option_left = matrix[i, j - 1] if j > 0 else np.inf
        move = np.argmin([option_diag, option_up, option_left])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    matches.append((0, 0))
    mappings_series_1[0].append(0)
    mappings_series_2[0].append(0)
    matches.reverse()
    for mp in mappings_series_1:
        mp.reverse()
    for mp in mappings_series_2:
        mp.reverse()

    return matches, matrix[-1, -1], mappings_series_1, mappings_series_2, matrix

def transform_step_token(
    base_model_tokenizer,
    base_model_input_ids,
    value_model_tokenizer,
    vocab_model_input_ids,
):
    """
    token alignment: use dtw to perform token alignment for two sequence.
    """
    base_model_tokens = base_model_tokenizer.convert_ids_to_tokens(base_model_input_ids)
    blending_model_tokens = value_model_tokenizer.convert_ids_to_tokens(
        vocab_model_input_ids
    )
    base_model_special_token = (
        TOKENIZER_TO_SPECIAL_TOKEN[base_model_tokenizer.__class__]
    )
    blending_model_special_token = (
        TOKENIZER_TO_SPECIAL_TOKEN[value_model_tokenizer.__class__]
    )

    def dist_fn(a, b):
        """calculate editdistance between two tokens, a is from blending model, b is from base model"""
        aa = a.replace(blending_model_special_token, "")
        bb = b.replace(base_model_special_token, "")
        w = 1
        if aa in bb or bb in aa:
            w = 0.1
        dist = editdistance.eval(aa, bb)
        return dist * w

    _, _, _, base_to_blending, _ = dtw(
        blending_model_tokens, base_model_tokens, norm_func=dist_fn
    )

    return (
        base_model_tokens,
        blending_model_tokens,
        base_model_special_token,
        blending_model_special_token,
        base_to_blending,
    )

with open(args.pure_text_file, "r", encoding="utf-8") as f:
    all_pure_text = json.load(f)

token_mapping_matrix = torch.zeros((len(base_vocab), len(value_vocab)))

def process_text(text):

    base_model_input_ids = base_tokenizer.encode(text, add_special_tokens=True)
    value_model_input_ids = value_tokenizer.encode(text, add_special_tokens=True)

    # Transform step token alignment

    base_tokens, blending_tokens, base_special_token, blending_special_token, base_to_blending = transform_step_token(
        base_model_tokenizer=base_tokenizer,
        base_model_input_ids=base_model_input_ids,
        value_model_tokenizer=value_tokenizer,
        vocab_model_input_ids=value_model_input_ids
    )

    if args.debug:

        # print("Base Tokens:", base_tokens)
        # print("Blending Tokens:", blending_tokens)
        # print("Base to Blending Mapping:", base_to_blending)

        for i, idx_list in enumerate(base_to_blending):
            print(f"Base token '{base_tokens[i]}' is mapped to blending tokens:", [blending_tokens[idx] for idx in idx_list])

    return base_model_input_ids, value_model_input_ids, base_to_blending


# system_prompt = """
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# """
# instruction = """
# Given that f(x) = 4x^3 - 9x - 14, find the value of f(2).
# """

# debug_text = f"""
# <s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{instruction}[/INST]"""

# print(process_text(debug_text))

if args.first_n:
    all_pure_text = all_pure_text[:args.first_n]

with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
    futures = {executor.submit(process_text, text): text for text in all_pure_text}
    results = []
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())

token_mapping_matrix = torch.zeros((len(base_vocab), len(value_vocab)))
for base_model_input_ids, value_model_input_ids, base_to_blending in tqdm(results):
    for i, base_id in enumerate(base_model_input_ids):
        for j in base_to_blending[i]:
            token_mapping_matrix[base_id, value_model_input_ids[j]] += 1

# Deal with the high frequecy no sense word
# 1. Three hot each row
# 2. spasify by coloum

temp_matrix_name = args.token_alignment_matrix_file.replace("_normalized", "")

torch.save(token_mapping_matrix, temp_matrix_name)

mask = (token_mapping_matrix != 0)

token_mapping_matrix = token_mapping_matrix - token_mapping_matrix.mean(dim=1, keepdim=True)

token_mapping_matrix = token_mapping_matrix * mask

row_maxs, _ = torch.max(token_mapping_matrix, dim=0, keepdim=True)
row_mins, _ = torch.min(token_mapping_matrix, dim=0, keepdim=True)

denominator = row_maxs - row_mins
denominator[denominator == 0] = 1

normalized_matrix = (token_mapping_matrix - row_mins) / denominator
normalized_matrix = normalized_matrix * mask

normalized_matrix = normalized_matrix * torch.zeros_like(normalized_matrix).scatter_(1, normalized_matrix.argmax(dim=1, keepdim=True), 1)

torch.save(normalized_matrix, args.token_alignment_matrix_file)