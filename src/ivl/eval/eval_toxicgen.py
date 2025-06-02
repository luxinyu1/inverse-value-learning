import argparse
import glob
import json
import os
import random
from collections import defaultdict

import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM

from .conversation import get_conv_template
from .inferencer import HFInferencer
from .modeling import LlamaResidualPlus, LlamaProxy, MixModelForLlama, MixModelCascadeForLlama, LlamaCascadeNoTextInputs, LlamaCascade, LlamaProbeForCausalLM

@torch.no_grad()
def score_generations(
    hate_classifier_tokenizer, hate_classifier_model, generations, batch_size=32
):
    classifications = []
    for i in trange(0, len(generations), batch_size):
        batch = generations[i: i + batch_size]
        inputs = hate_classifier_tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(hate_classifier_model.device)
        outputs = hate_classifier_model(**inputs)
        classifications.extend(outputs.logits.argmax(dim=-1).tolist())
    return classifications


def main(args):
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(os.path.join(args.output_dir, "toxicgen_metrics.json")):
        exit()

    # Load the testing data
    examples = []
    prompt_files = glob.glob(os.path.join(args.data_dir, "*.txt"))
    for task_file in tqdm(prompt_files, desc="Loading prompts"):
        with open(task_file, "r") as f:
            group_name = os.path.basename(task_file).split(".")[0]
            label = group_name.split("_")[0]
            minority_group = "_".join(group_name.split("_")[1:])
            group_prompts = [line.strip() for line in f]
            random.shuffle(group_prompts)
            group_prompts = group_prompts[:args.max_examples_per_group]
            for prompt in group_prompts:
                # minor cleaning: replace \\ with \
                prompt = prompt.replace("\\\\", "\\")
                prompt = prompt.replace("\\n", "\n")
                examples.append({
                    "text": prompt,
                    "label": label,
                    "target_groups": [minority_group],
                })

    prompts = []
    for example in examples:
        if args.use_chat_format:
            conv = get_conv_template(args.template_name)
            conv.system_message = " "
            conv.append_message("user", "Complete the following: " + example["text"])
            conv.append_message("model", None)
            prompt = conv.get_prompt()
        else:
            # we will use the original text from toxigen as the prompt.
            prompt = example["text"]
        prompts.append(prompt)

    with open(os.path.join(args.output_dir, "example_prompt_toxigen.txt"), "w") as fout:
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
            stop_strings="\n\n",
            do_sample=False,
        )
        print(output)
        print("=====================================")
        outputs.append(output)

    # Run the toxicity classifier
    toxigen_roberta_tokenizer = AutoTokenizer.from_pretrained("pretrained_models/toxigen_roberta")
    toxigen_roberta_classifier = AutoModelForSequenceClassification.from_pretrained(
        "pretrained_models/toxigen_roberta"
    )
    if torch.cuda.is_available():
        toxigen_roberta_classifier = toxigen_roberta_classifier.cuda()

    classifications = score_generations(
        toxigen_roberta_tokenizer,
        toxigen_roberta_classifier,
        outputs,
        batch_size=args.classifier_batch_size,
    )
    all_groups_scores = defaultdict(list)
    for example, output, classification in zip(examples, outputs, classifications):
        example["prediction"] = output
        example["is_toxic"] = classification
        for group in example["target_groups"]:
            all_groups_scores[group].append(classification)
    with open(os.path.join(args.output_dir, "toxicgen_predictions.jsonl"), "w") as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")

    # Calculate the performance
    performance = {}
    # calculate score by targeted group
    for group in all_groups_scores:
        performance[group] = sum(all_groups_scores[group]) / len(
            all_groups_scores[group]
        )
    # overall perf
    performance["overall"] = sum(classifications) / len(classifications)

    with open(os.path.join(args.output_dir, "toxicgen_metrics.json"), "w") as fout:
        print(f"Average Toxicity: {performance['overall']}")
        for group in performance:
            if group != "overall":
                print(f"Average Toxicity for {group}: {performance[group]}")
        json.dump(performance, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/eval/toxigen"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--classifier_batch_size",
        type=int,
        default=32,
        help="batch size to use for toxicity classifier.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--expert_model_name_or_path",
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
        "--system_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--use_template",
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
        "--max_examples_per_group",
        type=int,
        default=500,
        help="If given, we will only use this many instances per group. Default to 500 (half the available instances).",
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "residual", "cascade", "proxy", "cascade_nti", "linear", "mix"],
        default="single",
    )
    args = parser.parse_args()

    # assert args.base_model_name_or_path

    main(args)