
'''
This script is adapted from the official IFEVAL evaluation script:
https://github.com/google-research/google-research/tree/master/instruction_following_eval
'''

import argparse
import os

os.environ['NLTK_DATA'] = "./caches/nltk"

import re
import json
import torch
import random
import dataclasses
import collections
from typing import Dict, List, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from .conversation import get_conv_template

from .ifeval import instructions_registry
from .modeling import LlamaResidualPlus, LlamaProxy, MixModelForLlama, MixModelCascadeForLlama, LlamaCascadeNoTextInputs, LlamaCascade, LlamaProbeForCausalLM
from .inferencer import HFInferencer

@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: List[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: List[bool]


def read_prompt_list(input_jsonl_filename):
    """Read inputs from jsonl."""
    inputs = []
    with open(input_jsonl_filename, "r") as f:
        for l in f:
            example = json.loads(l)
            inputs.append(
                InputExample(key=example["key"],
                            instruction_id_list=example["instruction_id_list"],
                            prompt=example["prompt"],
                            kwargs=example["kwargs"]))
    return inputs


def write_outputs(output_jsonl_filename, outputs):
    """Writes outputs to jsonl."""
    assert outputs
    with open(output_jsonl_filename, "w") as f:
        for o in outputs:
            f.write(
                json.dumps(
                    {
                        attr_name: o.__getattribute__(attr_name)
                        for attr_name in [
                            name for name in dir(o) if not name.startswith("_")
                        ]
                    }
                )
            )
            f.write("\n")


def test_instruction_following_strict(
    inp,
    prompt_to_response,
):
    """Tests response to see if instrutions are followed."""
    response = prompt_to_response[inp.prompt]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    prompt_to_response,
):
    """Tests response for an upper bound for following instructions."""
    response = prompt_to_response[inp.prompt]
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def print_report(outputs):
    """Prints a report on accuracy scores."""

    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    tier0_total = collections.defaultdict(int)
    tier0_correct = collections.defaultdict(int)

    tier1_total = collections.defaultdict(int)
    tier1_correct = collections.defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(
            instruction_id_list, follow_instruction_list
        ):
            instruction_id = instruction_id.split(":")[0]
            tier0_total[instruction_id] += 1
            if followed_or_not:
                tier0_correct[instruction_id] += 1

        for instruction_id, followed_or_not in zip(
            instruction_id_list, follow_instruction_list
        ):
            tier1_total[instruction_id] += 1
            if followed_or_not:
                tier1_correct[instruction_id] += 1
            
    metrics = {
        "prompt-leval accuracy": prompt_correct / prompt_total,
        "instruction-level accuracy": instruction_correct / instruction_total,
        "tier0 accuracy": {instruction_id: tier0_correct[instruction_id] / tier0_total[instruction_id] for instruction_id in tier0_total},
        "tier1 accuracy": {instruction_id: tier1_correct[instruction_id] / tier1_total[instruction_id] for instruction_id in tier1_total},
    }

    print(json.dumps(metrics, indent=4))
    return metrics




def main(args):
    random.seed(42)

    if os.path.exists(os.path.join(args.output_dir, "ifeval_metrics.json")):
        exit()

    inputs = read_prompt_list(os.path.join(args.data_dir, "input_data.jsonl"))

    inputs = inputs[:args.max_num_examples]

    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "single":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name_or_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            load_in_8bit=args.load_in_8bit,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name_or_path
        )
    elif args.mode == "cascade":
        model = LlamaCascade.from_pretrained(
            model_name_or_path=args.guidance_model_name_or_path,
            pretrained_model_name_or_path=args.base_model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            load_in_8bit=args.load_in_8bit,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name_or_path
        )
    elif args.mode == "cascade_nti":
        model = LlamaCascadeNoTextInputs.from_pretrained(
            model_name_or_path=args.guidance_model_name_or_path,
            pretrained_model_name_or_path=args.base_model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            load_in_8bit=args.load_in_8bit,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name_or_path
        )
    elif args.mode == "residual":
        model = LlamaResidualPlus.from_pretrained(
            model_name_or_path=args.guidance_model_name_or_path,
            pretrained_model_name_or_path=args.base_model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            load_in_8bit=args.load_in_8bit,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name_or_path
        )
    elif args.mode == "linear":
        model = LlamaProbeForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.base_model_name_or_path,
            probe_name_or_path=args.guidance_model_name_or_path,
            torch_dtype=torch.bfloat16,
            load_in_8bit=args.load_in_8bit,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_name_or_path
        )
    elif args.mode == "mix":
        model = MixModelForLlama.from_pretrained(
            pretrained_model_name_or_path=args.base_model_name_or_path,
            model_name_or_path=args.guidance_model_name_or_path,
            alignment_matrix=args.alignment_matrix_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.guidance_model_name_or_path
        )
    else:
        raise NotImplementedError

    inferencer = HFInferencer(
        model=model,
        tokenizer=tokenizer
    )

    # prepare prompts    
    if args.use_chat_format:
        prompts = []
        for inp in inputs:
            conv = get_conv_template(args.template_name)
            conv.system_message = " "
            conv.append_message("user", inp.prompt)
            conv.append_message("model", None)
            prompts.append(
                conv.get_prompt()
            )
    else:
        prompts = [inp.prompt for inp in inputs]

    outputs = []
    for prompt in prompts:

        generation = inferencer.inference(
            prompt,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0,
        )

        generation = generation.rstrip(" </s>")
        outputs.append(generation)

    assert len(inputs) == len(outputs), "Number of inputs and outputs are not the same."
    response_dict = {inp.prompt: output for inp, output in zip(inputs, outputs)}

    # get instruction following results
    results = {}
    for eval_setup, func in [
        ("strict", test_instruction_following_strict),
        ("loose", test_instruction_following_loose),
    ]:
        print(f"Running {eval_setup} evaluation...")
        outputs = []
        for inp in inputs:
            outputs.append(func(inp, response_dict))
        follow_all_instructions = [o.follow_all_instructions for o in outputs]
        accuracy = sum(follow_all_instructions) / len(outputs)
        print("Accuracy: %f", accuracy)
        results[eval_setup] = {"Accuracy": accuracy}

        output_file_name = os.path.join(
            args.output_dir, f"ifeval_eval_results_{eval_setup}" + ".jsonl"
        )
        write_outputs(output_file_name, outputs)
        print(f"Results written to {output_file_name}")

        # Prints instruction following accuracy report.
        print("=" * 64)
        print(f"Detailed Scores:")
        detailed_scores = print_report(outputs)
        results[eval_setup].update(detailed_scores)

    # save the performance
    with open(os.path.join(args.output_dir, "ifeval_metrics.json"), "w") as fout:
        json.dump(results, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/eval/ifeval/"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/ifeval/"
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--guidance_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--alignment_matrix_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default=None, 
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--max_num_examples", 
        type=int, 
        default=None, 
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--template_name",
        type=str,
        default=None, 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "residual", "cascade", "proxy", "cascade_nti", "linear", "mix"],
        default="single",
    )

    args = parser.parse_args()

    main(args)
