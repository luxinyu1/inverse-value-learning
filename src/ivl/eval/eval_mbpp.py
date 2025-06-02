import argparse
import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['HF_ALLOW_CODE_EVAL'] = "1"

import json
import random
import torch
from datasets import load_dataset
from .codex_humaneval.data import write_jsonl
from .mbpp.evaluation import compute_code_eval
from .conversation import get_conv_template

from transformers import AutoModelForCausalLM, AutoTokenizer
from .modeling import LlamaResidualPlus, LlamaProxy, MixModelForLlama, MixModelCascadeForLlama, LlamaCascadeNoTextInputs, LlamaCascade, LlamaProbeForCausalLM
from .inferencer import HFInferencer
from tqdm import tqdm

def main(args):
    random.seed(42)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(os.path.join(args.output_dir, "mbpp_metrics.json")):
        exit()

    dataset = load_dataset("evalplus/mbppplus")['test']
    dataset.shuffle(seed=42)
    # Always head-out first 100 examples
    if args.max_num_examples is None:
        args.max_num_examples = len(dataset) - 100
    if args.max_num_examples > len(dataset) - 100:
        Warning("The number of examples is larger than the test set size. Will use the maximum number of examples.")
        args.max_num_examples = len(dataset) - 100
    test_data = dataset.select(range(100, min(100+args.max_num_examples, len(dataset))))
    print("Number of examples:", len(test_data))
    
    if args.use_chat_format:
        prompts = []
        answer = "Here is the completed function:\n\n\n```python\n"

        def apply_chat_format(tokenizer, inst, suffix):
            conv = get_conv_template(args.template_name)
            conv.system_message = " "
            conv.append_message("user", inst)
            conv.append_message("model", None)
            prompt = conv.get_prompt()
            prefix = "" if prompt[-1] in ["\n", " "] else " "
            return prompt + prefix + suffix
        
        if args.use_evalplus_prompt:
            instruction = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
            suffix = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:"
            for example in test_data:
                data_inst = instruction + f"\n```\n{example['prompt'].strip()}\n{random.choice(example['test_list'])}```\n"
                suffix_inst = f"\n{suffix}\n```python\n" + example['code'].split(":")[0] + ":"
                prompts.append((data_inst, suffix_inst)) 
        else:
            instruction = "Complete the following python function.\n\n\n"
            for example in test_data:
                prompts.append((instruction + example["prompt"] + example['code'].split(":")[0], answer))
    else:
        prompts = [example["prompt"] + example['code'].split(":")[0] for example in test_data]
    
    stop_sequences = ['```'] + args.additional_stop_sequence
    if args.use_evalplus_prompt:
        stop_sequences += ['\n"""', "\nassert", "\n#"]
    if args.template_name == "llama-3":
        stop_sequences += ['\n```\n\n']

    results_file = None

    if os.path.isfile(os.path.join(args.output_dir, 'mbpp_chat_predictions.jsonl')):
        results_file = os.path.join(args.output_dir, 'mbpp_chat_predictions.jsonl')
    elif os.path.isfile(os.path.join(args.output_dir, 'mbpp_predictions.jsonl')):
        results_file = os.path.join(args.output_dir, 'mbpp_predictions.jsonl')
    else:
        pass
        
    if results_file is None:
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
                probe_name_or_path=args.guidance_model_name_or_path,
                pretrained_model_name_or_path=args.base_model_name_or_path,
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
        
        if args.use_chat_format:
            prompts = [apply_chat_format(tokenizer, inst, suffix) for (inst, suffix) in prompts]

        # Because many tokenizers will treat the word after space differently from the original word alone, 
        # to be consistent, we add a space before tokenization and remove it after tokenization.
        # stop_sequences = [tokenizer.encode(" " + x, add_special_tokens=False)[1:] for x in stop_sequences]
        outputs_per_sampling_iter = []
        for sampling_iter in range(args.unbiased_sampling_size_n):
            print(f"Sampling iter: {sampling_iter} / {args.unbiased_sampling_size_n}")
            sampling_outputs = []
            for prompt in prompts:
                generation = inferencer.inference(
                    prompt,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    stop_strings=stop_sequences,
                    do_sample=True,  # if only pass@1 is evaluated, we do greedy decoding.
                    temperature=args.temperature,
                )
                for ss in stop_sequences:
                    generation = generation.rstrip(ss) # strip the stop sequence
                sampling_outputs.append(generation)
            outputs_per_sampling_iter.append(sampling_outputs)
        # regroup the outputs to match the number of test data.
        outputs = []
        for i in range(len(prompts)):
            for j in range(args.unbiased_sampling_size_n):
                outputs.append(outputs_per_sampling_iter[j][i])
    else:
        with open(results_file, "r") as f:
            outputs = [json.loads(line)['completion'] for line in f]

    # duplicates test data to match the number of outputs.
    duplicate_test_data = [
        example for example in test_data for _ in range(args.unbiased_sampling_size_n)
    ]
    duplicate_prompts = [
        prompt for prompt in prompts for _ in range(args.unbiased_sampling_size_n)
    ]
    # if evalplus setup, we have to re-add the code prefix to the output.
    if args.use_evalplus_prompt:
        predictions = [{"task_id": example["task_id"], "prompt": prompt, "completion": example['code'].split(":")[0] + ":" + output, "test_cases": example['test']} 
                    for example, prompt, output in zip(duplicate_test_data, duplicate_prompts, outputs)]
    else:
        predictions = [{"task_id": example["task_id"], "prompt": prompt, "completion": output.strip("```"), "test_cases": example['test']} 
                   for example, prompt, output in zip(duplicate_test_data, duplicate_prompts, outputs)]
    predictions_noresult = [{"task_id":pred["task_id"], "prompt":pred['prompt'], "completion": pred['completion']} for pred in predictions]
    if args.use_chat_format:
        prediction_save_path = os.path.join(args.output_dir, "mbpp_chat_predictions.jsonl")
    else:
        prediction_save_path = os.path.join(args.output_dir, "mbpp_predictions.jsonl")
    write_jsonl(prediction_save_path, predictions_noresult)
    pass_at_k_results, results = compute_code_eval(
        predictions=predictions,
        k=args.eval_pass_at_ks,
        num_workers=64,
        timeout=10.0
    )

    print(pass_at_k_results)

    with open(os.path.join(args.output_dir, "mbpp_metrics.json"), "w") as fout:
        json.dump(pass_at_k_results, fout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_name_or_path", 
        type=str, 
        default=None, 
        help="If specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default=None, 
        help="If specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--alignment_matrix_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="results/codex_eval", 
        help="Directory to save the results."
    )
    parser.add_argument(
        "--guidance_model_name_or_path",
        type=str,
    )
    parser.add_argument(
        "--max_num_examples",
        type=int,
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="Batch size for evaluation."
    )
    parser.add_argument(
        "--eval_pass_at_ks", 
        nargs="+", 
        type=int, 
        default=[1], 
        help="Multiple k's that we will report pass@k."
    )
    parser.add_argument(
        "--unbiased_sampling_size_n", 
        type=int, 
        default=20,
        help="Codex HumanEval requires `n` sampled generations per prompt, to estimate the unbiased pass@k. "
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for sampling. This is should be low for evaluating smaller pass@k, and high for larger pass@k."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="Load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--template_name",
        type=str,
        default=None, 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        '--additional_stop_sequence',
        type=str,
        nargs="+",
        default=[],
        help="Additional stop sequences to use when generating completions. Useful for e.g. llama-3-instruct."
    )
    parser.add_argument(
        '--use_evalplus_prompt',
        action="store_true",
        help="If given, we will use the evalplus prompting setup, to better match scores on the evalplus leaderboard."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "residual", "cascade", "proxy", "cascade_nti", "linear", "mix"],
        default="single",
    )
    parser.add_argument("--results_file", type=str)
    args = parser.parse_args()
    # model_name_or_path and openai_engine cannot be both None or both not None.
    assert args.unbiased_sampling_size_n >= max(args.eval_pass_at_ks), "n should be larger than the largest k in eval_pass_at_ks."
    main(args)