CUDA_VISIBLE_DEVICES=0 python ./src/ivl/serve/serve_llama.py --model-name "llama_residual_plus" \
                                --model-path "saved_models/llama_2-7b_residual_plus_norm_for_inference" \
                                --pretrained-model-path "pretrained_models/Llama-2-7b-hf" \