#!/bin/bash
#/data/louhaoran/checkpoints/llava-1.5v-7b-finetune-lora-336-tfm-adaptive-pooling/
#/data/louhaoran/checkpoints/llava-1.5v-7b-finetune-lora-336-conv336-crop-fuse-25/
#/data/louhaoran/checkpoints/llava-1.5v-7b-finetune-lora-336-conv336-adaptive-pooling-fuse-25/
python -m llava.eval.model_vqa \
    --model-base /home/louhaoran/MyLLM/ckpt/vicuna-7b-v1.5/ \
    --model-path /data/louhaoran/checkpoints/llava-1.5v-7b-finetune-lora-336-conv336-adaptive-pooling-fuse-25/ \
    --question-file /data/louhaoran/llava_bench/questions.jsonl \
    --image-folder /data/louhaoran/llava_bench_image/images/ \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/llava-1.5v-7b-finetune-lora-336-conv336-adaptive-pooling-fuse-25.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

#mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews
#
#
#python llava/eval/eval_gpt_review_bench.py \
#    --question /data/lyc/llava/eval/llava-bench-in-the-wild/questions.jsonl \
#    --context /data/lyc/llava/eval/llava-bench-in-the-wild/context.jsonl \
#    --rule llava/eval/table/rule.json \
#    --answer-list \
#        /data/lyc/llava/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
#        /home/louhaoran/LLava+/playground/data/eval/llava-bench-in-the-wild/answers/llava-1.5v-7b-finetune-lora-336-tfm-adaptive-pooling.jsonl\
#    --output \
#        playground/data/eval/llava-bench-in-the-wild/reviews/llava-1.5v-7b-finetune-lora-336-tfm-adaptive-pooling-gpt4o.jsonl
#
#python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/llava-1.5v-7b-finetune-lora-336-tfm-adaptive-pooling-gpt4o.jsonl
