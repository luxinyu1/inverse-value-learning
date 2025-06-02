export MODEL_NAME=

python ./src/ivl/eval/eval_mt_bench.py  --model-path ${MODEL_NAME} \
                                --model-id "llama-2" \
                                --answer-file "./outputs/mt_bench/${MODEL_NAME}.jsonl" \
                                --system-message " "