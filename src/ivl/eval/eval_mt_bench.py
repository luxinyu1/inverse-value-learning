import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype

from modeling import LlamaResidualPlus
from transformers import AutoModelForCausalLM, AutoTokenizer
from inferencer import HFInferencer

def run_eval(
    residual_model,
    model_path,
    pretrained_model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    system_message,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    if residual_model and model_id.startswith("llama"):
        assert pretrained_model_path is not None
        model = LlamaResidualPlus.from_pretrained(
            model_name_or_path=model_path,
            pretrained_model_name_or_path=pretrained_model_path
        )
        model.to_cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    inferencer = HFInferencer(
        model,
        tokenizer
    )

    get_answers_func = get_model_answers

    ans_handles = []
    for i in tqdm(range(0, len(questions))):
        ans_handles.append(
            get_answers_func(
                inferencer,
                model_id,
                questions[i],
                answer_file,
                max_new_token,
                num_choices,
                system_message,
            )
        )

@torch.inference_mode()
def get_model_answers(
    inferencer,
    model_id,
    question,
    answer_file,
    max_new_token,
    num_choices,
    system_message,
):

    if question["category"] in temperature_config:
        temperature = temperature_config[question["category"]]
    else:
        temperature = 0.7

    print("temperature: ", temperature)

    tokenizer = inferencer.get_tokenizer()

    choices = []
    for i in range(num_choices):
        torch.manual_seed(i)
        conv = get_conversation_template(model_id)
        if system_message is not None:
            conv.system_message = system_message
        turns = []
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            if temperature < 1e-4:
                do_sample = False
            else:
                do_sample = True

            print("Input:", prompt)

            # some models may error out when generating long outputs
            try:
                if do_sample==False:
                    output = inferencer.inference(
                        prompt,
                        temperature=0,
                        do_sample=do_sample,
                        max_new_tokens=max_new_token
                    )
                else:
                    output = inferencer.inference(
                        prompt,
                        temperature=temperature,
                        do_sample=do_sample,
                        max_new_tokens=max_new_token
                    )

                if conv.name.startswith("qwen"):
                    output = output[: output.find(conv.sep)]
                else:
                    if conv.stop_str and isinstance(conv.stop_str, list):
                        stop_str_indices = sorted(
                            [
                                output.find(stop_str)
                                for stop_str in conv.stop_str
                                if output.find(stop_str) > 0
                            ]
                        )
                        if len(stop_str_indices) > 0:
                            output = output[: stop_str_indices[0]]
                    elif conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]

                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()
            except RuntimeError as e:
                print("ERROR question ID: ", question["question_id"])
                output = "ERROR"

            print("Output:", output)

            print("======================")

            conv.update_last_message(output)
            turns.append(output)

        choices.append({"index": i, "turns": turns})

    # Dump answers
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(os.path.expanduser(answer_file), "a") as fout:
        ans_json = {
            "question_id": question["question_id"],
            "answer_id": shortuuid.uuid(),
            "model_id": model_id,
            "choices": choices,
            "tstamp": time.time(),
        }
        fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--residual-model",
        action="store_true"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--pretrained-model-path",
        type=str,
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--inferencer",
        choices=["guidance", "default", "proxy"],
        default="guidance",
    )
    parser.add_argument(
        "--system-message",
        default=None,
    )

    args = parser.parse_args()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        residual_model=args.residual_model,
        model_path=args.model_path,
        pretrained_model_path=args.pretrained_model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        system_message=args.system_message,
    )

    reorg_answer_file(answer_file)
