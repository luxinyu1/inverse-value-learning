export HF_DATASETS_CACHE="./caches/hf_cache/datasets"
export HF_ENDPOINT="https://hf-mirror.com"

MODEL_NAME="1_1B_1_1B"
MODEL_PATH="saved_models/llama_residual_plus_for_inference"
PRETRAINED_MODLE_PATH="pretrained_models/TinyLlama-1.1B-intermediate-step-1431k-3T"

FINAL_MODEL_PATH="${MODEL_PATH}|${PRETRAINED_MODLE_PATH}"

echo ${FINAL_MODEL_PATH}

accelerate launch -m lm_eval --model custom \
    --model_args pretrained="${FINAL_MODEL_PATH}" \
    --tasks "arc_challenge" \
    --batch_size auto \
    --num_fewshot 25 \
    --output_path "./outputs/${MODEL_NAME}/arc_challenge.json" \

accelerate launch -m lm_eval --model custom \
    --model_args pretrained="${FINAL_MODEL_PATH}" \
    --tasks "hellaswag" \
    --batch_size auto \
    --num_fewshot 10 \
    --output_path "./outputs/${MODEL_NAME}/hellaswag.json" \

accelerate launch -m lm_eval --model custom \
    --model_args pretrained="${FINAL_MODEL_PATH}" \
    --tasks "truthfulqa" \
    --batch_size auto \
    --num_fewshot 0 \
    --output_path "./outputs/${MODEL_NAME}/truthfulqa_mc.json" \

accelerate launch -m lm_eval --model custom \
    --model_args pretrained="${FINAL_MODEL_PATH}" \
    --tasks "gsm8k" \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path "./outputs/${MODEL_NAME}/gsm8k.json" \

accelerate launch -m lm_eval --model custom \
    --model_args pretrained="${FINAL_MODEL_PATH}" \
    --tasks "mmlu" \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path "./outputs/${MODEL_NAME}/mmlu.json" \

accelerate launch -m lm_eval --model custom \
    --model_args pretrained="${FINAL_MODEL_PATH}" \
    --tasks "winogrande" \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path "./outputs/${MODEL_NAME}/winogrande.json"