MODEL_NAME="1_1B_7B"
BASE_MODEL="pretrained_models/Llama-2-7b-hf"
GUIDANCE_MODEL="saved_models/tinyllama_residual_plus_for_inference"
TEMPLATE="llama-2"

python -m src.ivl.eval.eval_toxicgen --mode residual \
    --model_name ${MODEL_NAME} \
    --base_model_name_or_path ${BASE_MODEL} \
    --guidance_model_name_or_path ${GUIDANCE_MODEL} \
    --output_dir outputs/${MODEL_NAME} \
    --max_examples_per_group 200 \
    --template_name ${TEMPLATE} \
    --use_chat_format \

python -m src.ivl.eval.eval_gsm8k --mode residual \
    --model_name ${MODEL_NAME} \
    --base_model_name_or_path ${BASE_MODEL} \
    --guidance_model_name_or_path ${GUIDANCE_MODEL} \
    --output_dir outputs/${MODEL_NAME} \
    --template_name ${TEMPLATE} \
    --use_chat_format \

python -m src.ivl.eval.eval_ifeval \
    --mode residual \
    --data_dir data/eval/ifeval/ \
    --output_dir outputs/${MODEL_NAME} \
    --base_model_name_or_path ${BASE_MODEL} \
    --guidance_model_name_or_path ${GUIDANCE_MODEL} \
    --tokenizer ${BASE_MODEL} \
    --use_chat_format \
    --template_name ${TEMPLATE} \

python -m src.ivl.eval.eval_mbpp \
    --mode residual \
    --use_chat_format \
    --template_name ${TEMPLATE} \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --output_dir outputs/${MODEL_NAME} \
    --base_model_name_or_path ${BASE_MODEL} \
    --guidance_model_name_or_path ${GUIDANCE_MODEL} \
    --tokenizer ${BASE_MODEL} \

python -m src.ivl.eval.eval_humaneval \
    --mode residual \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --data_file_hep data/eval/codex_humaneval/humanevalpack.jsonl  \
    --use_chat_format \
    --template_name ${TEMPLATE} \
    --eval_pass_at_ks 1 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --output_dir outputs/${MODEL_NAME} \
    --base_model_name_or_path ${BASE_MODEL} \
    --tokenizer ${BASE_MODEL} \
    --guidance_model_name_or_path ${GUIDANCE_MODEL} \