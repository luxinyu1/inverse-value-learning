cd ./data/
 
wget https://lxylab.oss-cn-shanghai.aliyuncs.com/ivl/infinite_7m.json
wget https://lxylab.oss-cn-shanghai.aliyuncs.com/ivl/merged_en_zh-split.json

cd ..

mkdir -p ./pretrained_models

cd pretrained_models
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
export HF_ENDPOINT=https://hf-mirror.com
./hfd.sh TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T -x 10
./hfd.sh meta-llama/Llama-2-7b-hf -x 10 --hf_username your_name --hf_token your_token # add your hf token here