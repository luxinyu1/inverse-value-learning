export NCCL_CUMEM_ENABLE=0
export HF_ENDPOINT="https://hf-mirror.com"

export HF_DATASETS_CACHE="./caches/hf_cache/datasets"

MODEL_NAME="TinyLlama-1.1B-intermediate-step-1431k-3T"
MODEL_PATH="pretrained_models/TinyLlama-1.1B-intermediate-step-1431k-3T"

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "arc_challenge" \
    --batch_size auto \
    --num_fewshot 25 \
    --output_path "./outputs/${MODEL_NAME}/arc_challenge.json" \

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "hellaswag" \
    --batch_size auto \
    --num_fewshot 10 \
    --output_path "./outputs/${MODEL_NAME}/hellaswag.json" \

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "truthfulqa" \
    --batch_size auto \
    --num_fewshot 0 \
    --output_path "./outputs/${MODEL_NAME}/truthfulqa_mc.json" \

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "gsm8k" \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path "./outputs/${MODEL_NAME}/gsm8k.json" \

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "mmlu" \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path "./outputs/${MODEL_NAME}/mmlu.json" \

accelerate launch -m lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "winogrande" \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path "./outputs/${MODEL_NAME}/winogrande.json"