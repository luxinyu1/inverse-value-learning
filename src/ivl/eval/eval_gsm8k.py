
import argparse
import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

import re
import json
import random
import evaluate
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from .conversation import get_conv_template
from .inferencer import HFInferencer
from .modeling import LlamaResidualPlus, LlamaProxy, MixModelForLlama, MixModelCascadeForLlama, LlamaCascadeNoTextInputs, LlamaCascade, LlamaProbeForCausalLM


exact_match = evaluate.load("exact_match")


def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output


def main(args):
    random.seed(42)

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(os.path.join(args.output_dir, "gsm8k_metrics.json")):
        exit()

    print("Loading data...")
    test_data = []
    with open(os.path.join(args.data_dir, "test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append({
                "question": example["question"],
                "answer": example["answer"].split("####")[1].strip()
            })

    # some numbers are in the `x,xxx` format, and we want to remove the comma
    for example in test_data:
        example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
        assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"

    if args.max_examples and len(test_data) > args.max_examples:
        test_data = random.sample(test_data, args.max_examples)

    prompt_prefix = "Answer the following question.\n\n"

    prompts = []
    for example in test_data:
        prompt = prompt_prefix + "Question: " + example["question"].strip()
        if args.use_chat_format:
            conv = get_conv_template(args.template_name)
            conv.system_message = " "
            conv.append_message("user", prompt)
            conv.append_message("model", None)
            prompt = conv.get_prompt()
            if prompt[-1] in ["\n", " "]:
                prompt += "Answer:"
            else:
                prompt += " Answer:"
        else:
            prompt += "\nAnswer:"
        prompts.append(prompt)

    with open(os.path.join(args.output_dir, "example_prompt_gsm8k.txt"), "w") as fout:
        print(prompts[0])
        fout.write(prompts[0])

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

    outputs = []
    for p in tqdm(prompts):
        print(p)
        output = inferencer.inference(
            p,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=False,
        )
        print(output)
        print("=====================================")
        outputs.append(output)

    outputs = [trim_output(o) for o in outputs]

    predictions = []
    for output in outputs:
        # replace numbers like `x,xxx` with `xxxx`
        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            predictions.append(numbers[-1])
        else:
            predictions.append(output)

    print("Calculating accuracy...")
    targets = [example["answer"] for example in test_data]

    em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match : {em_score}")

    predictions = [{
        "question": example["question"],
        "answer": example["answer"],
        "model_output": output,
        "prediction": pred
    } for example, output, pred in zip(test_data, outputs, predictions)]

    with open(os.path.join(args.output_dir, "gsm8k_predictions.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")

    with open(os.path.join(args.output_dir, "gsm8k_metrics.json"), "w") as fout:
        json.dump({
            "exact_match": em_score
        }, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/eval/gsm"
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
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
        "--base_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-13b-hf',
    )
    parser.add_argument(
        "--expert_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-7b-chat-hf',
    )
    parser.add_argument(
        "--guidance_model_name_or_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true"
    )
    parser.add_argument(
        "--template_name",
        type=str,
        default=None, 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "residual", "cascade", "proxy", "cascade_nti", "linear", "mix"],
        default="single",
    )
    args = parser.parse_args()

    main(args)
