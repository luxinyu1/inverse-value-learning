import torch
from safetensors import safe_open
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.tokenization_llama import LlamaTokenizer
import glob
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", type=str, required=True)
parser.add_argument("--tokenizer-dir", type=str, required=True)
parser.add_argument("--output-model-dir", type=str, required=True)
args = parser.parse_args()

tensors_path = glob.glob(os.path.join(args.model_dir, "*.safetensors"))

state_dict = {}
for path in tensors_path:
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith('pretrained_model'):
                continue
            elif key.startswith('model'):
                parts = key.split('.')
                parts = parts[1:]
                new_key = '.'.join(parts)
                state_dict[new_key] = f.get_tensor(key)
            else:
                print(key)

print(state_dict.keys())

config = LlamaConfig.from_pretrained(args.tokenizer_dir)

tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_dir)

model = LlamaForCausalLM(config)

model.load_state_dict(state_dict=state_dict)

model.save_pretrained(args.output_model_dir)
tokenizer.save_pretrained(args.output_model_dir)