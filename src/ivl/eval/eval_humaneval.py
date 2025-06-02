
import argparse
import os
import json
import random
import torch
from tqdm import tqdm
from .codex_humaneval.data import write_jsonl, read_problems
from .codex_humaneval.evaluation import evaluate_functional_correctness

from transformers import AutoModelForCausalLM, AutoTokenizer
from .conversation import get_conv_template

from .modeling import LlamaResidualPlus, LlamaProxy, MixModelForLlama, MixModelCascadeForLlama, LlamaCascadeNoTextInputs, LlamaCascade, LlamaProbeForCausalLM

from .inferencer import HFInferencer

def main(args):
    random.seed(42)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(os.path.join(args.output_dir, "humaneval_metrics.json")):
        exit()

    test_data = list(read_problems(args.data_file).values())
    if args.max_num_examples is not None and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)
    print("Number of examples:", len(test_data))

    # these stop sequences are those mentioned in the codex paper.
    stop_sequences = ["\nclass", "\ndef", "\n#", "\nif", "\nprint"] + args.additional_stop_sequence

    os.makedirs(args.output_dir, exist_ok=True)

    prediction_save_path = os.path.join(args.output_dir, "humaneval_predictions.jsonl")

    if not os.path.exists(prediction_save_path):
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
            prompts = []

            # If available use more realistic instructions from HumanEvalPack (https://hf.co/datasets/bigcode/humanevalpack)
            if os.path.exists(args.data_file_hep):
                with open(args.data_file_hep, "r") as f:
                    instructions = [json.loads(l) for l in f]
                    instructions_dict = {
                        x["task_id"].replace("Python", "HumanEval"): x["instruction"] for x in instructions
                    }
                answer = "Here is the function:\n\n```python\n"
                stop_sequences.append("\n```")
            else:
                print(f"Could not find HumanEvalPack file at {args.data_file_hep}, which will result in significantly worse performance. You can download it at https://hf.co/datasets/bigcode/humanevalpack/blob/main/data/python/data/humanevalpack.jsonl")
                instructions_dict = None
                answer = "Here is the completed function:\n\n\n```python\n"
                stop_sequences.append("\n```")

            def apply_chat_format(tokenizer, inst, suffix):
                conv = get_conv_template(args.template_name)
                conv.system_message = " "
                conv.append_message("user", inst)
                conv.append_message("model", None)
                prompt = conv.get_prompt()
                prefix = "" if prompt[-1] in ["\n", " "] else " "
                return prompt + prefix + suffix
                
            instruction = "Complete the following python function.\n\n\n"
            for example in test_data:
                if instructions_dict is not None:
                    instruction = instructions_dict[example["task_id"]]
                    prompts.append((instruction, answer + example["prompt"]))
                else:
                    prompts.append((instruction + example["prompt"], answer))   

            prompts = [apply_chat_format(tokenizer, inst, suffix) for (inst, suffix) in prompts]
            
        else:
            prompts = [example["prompt"] for example in test_data]

        outputs_per_sampling_iter = []
        for sampling_iter in range(args.unbiased_sampling_size_n):
            print(f"Sampling iter: {sampling_iter} / {args.unbiased_sampling_size_n}")
            sampling_outputs = []
            for prompt in tqdm(prompts):
                generation = inferencer.inference(
                    prompt,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    stop_strings=stop_sequences,
                    num_return_sequences=1,  # we don't use the hf num_return_sequences, because otherwise the real batch size will be multiplied by it and often cause oom.
                    do_sample=True,  # if only pass@1 is evaluated, we do greedy decoding.
                    top_p=0.95,
                    temperature=args.temperature,
                )

                for ss in stop_sequences:
                    generation = generation.rstrip(ss) # strip the stop sequence
                
                if not generation.startswith('    '):
                    spaces_to_add = 4 - (len(generation) - len(generation.lstrip()))
                    if spaces_to_add > 0:
                        generation = ' ' * spaces_to_add + generation
                
                sampling_outputs.append(generation)

                print(generation)

            outputs_per_sampling_iter.append(sampling_outputs)
        # regroup the outputs to match the number of test data.
        outputs = []
        for i in range(len(prompts)):
            for j in range(args.unbiased_sampling_size_n):
                outputs.append(outputs_per_sampling_iter[j][i])

        # duplicates test data to match the number of outputs.
        duplicate_test_data = [
            example for example in test_data for _ in range(args.unbiased_sampling_size_n)
        ]
        assert len(duplicate_test_data) == len(outputs)
        predictions = [{"task_id": example["task_id"], "prompt": example["prompt"], "completion": output} for example, output in zip(duplicate_test_data, outputs)]

        write_jsonl(prediction_save_path, predictions)

    pass_at_k_results = evaluate_functional_correctness(
        sample_file=prediction_save_path,
        k=args.eval_pass_at_ks,
        problems={example["task_id"]: example for example in test_data},
        n_workers=64
    )

    with open(os.path.join(args.output_dir, "humaneval_metrics.json"), "w") as fout:
        json.dump(pass_at_k_results, fout)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", 
        type=str, 
        default="data/codex_eval/HumanEval.jsonl.gz",
        help="Path to the HumanEval data file."
    )
    parser.add_argument(
        "--data_file_hep", 
        type=str, 
        default="data/codex_eval/humanevalpack.jsonl",
        help="Path to the HumanEvalPack data file."
    )
    parser.add_argument(
        "--max_num_examples", 
        type=int, 
        default=None,
        help="Maximum number of examples to evaluate."
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
        "--output_dir", 
        type=str, 
        default="results/codex_eval", 
        help="Directory to save the results."
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
    parser.add_argument(
        '--additional_stop_sequence',
        type=str,
        nargs="+",
        default=[],
        help="Additional stop sequences to use when generating completions. Useful for e.g. llama-3-instruct."
    )
    args = parser.parse_args()
    # model_name_or_path and openai_engine cannot be both None or both not None.

    main(args)
