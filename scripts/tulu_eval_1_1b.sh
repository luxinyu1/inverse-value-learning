MODEL_NAME="TinyLlama-1.1B-intermediate-step-1431k-3T"
BASE_MODEL="pretrained_models/TinyLlama-1.1B-intermediate-step-1431k-3T"

python -m src.ivl.eval.eval_toxicgen --mode single \
    --model_name ${MODEL_NAME} \
    --base_model_name_or_path ${BASE_MODEL} \
    --output_dir outputs/${OUTPUT_DIR} \
    --max_examples_per_group 200 \

python -m src.ivl.eval.eval_gsm8k --mode single \
    --model_name ${MODEL_NAME} \
    --base_model_name_or_path ${BASE_MODEL} \
    --output_dir outputs/${MODEL_NAME} \

python -m src.ivl.eval.eval_ifeval \
    --mode single \
    --data_dir data/eval/ifeval/ \
    --output_dir outputs/${MODEL_NAME} \
    --base_model_name_or_path ${BASE_MODEL} \
    --tokenizer ${BASE_MODEL} \

python -m src.ivl.eval.eval_mbpp \
    --mode single \
    --eval_pass_at_ks 1 5 10 20 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --output_dir outputs/${MODEL_NAME} \
    --base_model_name_or_path ${BASE_MODEL} \
    --tokenizer ${BASE_MODEL} \

python -m src.ivl.eval.eval_humaneval \
    --mode single \
    --data_file data/eval/codex_humaneval/HumanEval.jsonl.gz  \
    --data_file_hep data/eval/codex_humaneval/humanevalpack.jsonl  \
    --eval_pass_at_ks 1 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --output_dir outputs/${MODEL_NAME} \
    --base_model_name_or_path ${BASE_MODEL} \
    --tokenizer ${BASE_MODEL} \