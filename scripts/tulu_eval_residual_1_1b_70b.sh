python -m src.ivl.eval.eval_toxicgen --mode "residual" \
    --model_name "1_1B_70B" \
    --base_model_name_or_path "pretrained_models/hf_llama-2/70B" \
    --guidance_model_name_or_path "saved_models/tinyllama_residual_plus_for_inference" \
    --output_dir "outputs/1_1B_70B" \
    --max_examples_per_group 200 \
    --template_name "llama-2" \
    --use_chat_format \

python -m src.ivl.eval.eval_gsm8k --mode "residual" \
    --model_name "1_1B_70B" \
    --base_model_name_or_path "pretrained_models/hf_llama-2/70B" \
    --guidance_model_name_or_path "saved_models/tinyllama_residual_plus_for_inference" \
    --output_dir "outputs/1_1B_70B" \
    --template_name "llama-2" \
    --use_chat_format \

python -m src.ivl.eval.eval_ifeval \
    --mode "residual" \
    --data_dir data/eval/ifeval/ \
    --output_dir "outputs/1_1B_70B" \
    --base_model_name_or_path "pretrained_models/hf_llama-2/70B" \
    --guidance_model_name_or_path "saved_models/tinyllama_residual_plus_for_inference" \
    --tokenizer "pretrained_models/hf_llama-2/70B" \
    --use_chat_format \
    --template_name "llama-2" \

python -m src.ivl.eval.eval_mbpp \
    --mode "residual" \
    --use_chat_format \
    --template_name "llama-2" \
    --eval_pass_at_ks 1 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --output_dir "outputs/1_1B_70B" \
    --base_model_name_or_path "pretrained_models/hf_llama-2/70B" \
    --guidance_model_name_or_path "saved_models/tinyllama_residual_plus_for_inference" \
    --tokenizer "pretrained_models/hf_llama-2/70B" \

python -m src.ivl.eval.eval_humaneval \
    --mode "residual" \
    --data_file "data/eval/codex_humaneval/HumanEval.jsonl.gz"  \
    --data_file_hep "data/eval/codex_humaneval/humanevalpack.jsonl"  \
    --use_chat_format \
    --template_name "llama-2" \
    --eval_pass_at_ks 1 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --output_dir "outputs/1_1B_70B" \
    --base_model_name_or_path "pretrained_models/hf_llama-2/70B" \
    --tokenizer "pretrained_models/hf_llama-2/70B" \
    --guidance_model_name_or_path "saved_models/tinyllama_residual_plus_for_inference" \