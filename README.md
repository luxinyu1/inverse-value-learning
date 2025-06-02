This repository contains the code and datasets used in the paper titled "Transferable Post-training via Inverse Value Learning".

## Install

```
conda create -n ivl python=3.10
conda activate ivl
pip install -e .
```

## Data Preparation

Download the necessary data using the provided script:

```bash
bash ./scripts/download.sh
```
| File Name | Description |
| ------- | ------- |
| ```data/merged_en_zh-split.json``` | ShareGPT Data |
| ```data/infinite_7m.json``` | InfinityInstruct Data |

## Train

All the related training scripts are provided in ```./scripts/finetune_*.sh```. For example, to finetune a 1.1B value model, you can try:

```bash

bash ./scripts/finetune_tinyllama_residual_plus.sh

bash ./scripts/finetune_tinyllama_residual_plus_norm.sh # with additional norm term

```

After training, you should run the weight converting scripts for further inference.

```bash
bash ./scripts/convert_model_for_inference.sh
```

## Eval

To execute the zero-shot evaluation, run:

```
bash ./scripts/tulu_eval_residual_*.sh
```

To execute the few-shot evaluation, you should first install the modified version of ```lm-eval```.

```bash
conda create -n lm-eval-ivl python=3.10
conda activate lm-eval-ivl
pip install lm-evaluation-harness.zip
bash scripts/lm_eval_custom_7b.sh
```

## Serve

We also provide a serve interface for inverse value learning. Try it by executing:

```bash
bash ./scripts/serve.sh
```

## Citation

Please cite our work if you find our code useful:

```
@inproceedings{lu-etal-2025-transferable,
    title = "Transferable Post-training via Inverse Value Learning",
    author = "Lu, Xinyu  and
      Wen, Xueru  and
      Lu, Yaojie  and
      Yu, Bowen  and
      Lin, Hongyu  and
      Yu, Haiyang  and
      Sun, Le  and
      Han, Xianpei  and
      Li, Yongbin",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.227/",
    pages = "4436--4447",
    ISBN = "979-8-89176-189-6",
    abstract = "As post-training processes utilize increasingly large datasets and base models continue to grow in size, the computational demands and implementation challenges of existing algorithms are escalating significantly. In this paper, we propose modeling the changes at the logits level during post-training using a separate neural network (i.e., the value network). After training this network on a small base model using demonstrations, this network can be seamlessly integrated with another pre-trained models during inference, enabling them to achieve similar capability enhancements. We systematically investigate the best practices for this paradigm in terms of pre-training weights and connection schemes. We demonstrate that the resulting value network has broad transferability across pre-trained models of different parameter sizes within the same family, models undergoing continuous pre-training within the same family, and models with different vocabularies across families. In certain cases, it can achieve performance comparable to full-parameter fine-tuning. Furthermore, we explore training methods to enhance transferability, which effectively improve the transfer performance of the value model across models of various parameter scales and prevent overfitting to the base model used during training."
}
```