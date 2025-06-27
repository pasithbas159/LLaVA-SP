#!/bin/bash

SPLIT="mmbench_dev_20230712"

#/data/lyc/llava/eval/mmbench/
#/data/louhaoran/checkpoints/llava-1.5v-7b-finetune-lora-336-conv336-adaptive-pooling-fuse-25/
#/data/louhaoran/llava-v1.6-vicuna-7b/
python -m llava.eval.model_vqa_mmbench \
    --model-base /home/louhaoran/MyLLM/ckpt/vicuna-7b-v1.5/ \
    --model-path /data/louhaoran/checkpoints/llava-1.5v-7b-finetune-lora-336-conv336-adaptive-pooling-fuse-25/ \
    --question-file /data/lyc/llava/eval/mmbench//$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-v1.6-vicuna-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-13b
