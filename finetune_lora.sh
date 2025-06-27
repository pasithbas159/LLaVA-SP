#!/bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 25751 train.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /root/LLaVA-SP/scripts/zero2.json \
    --model_name_or_path /datadisk/llm/vicuna-13b-v1.5/ \
    --version v1 \
    --data_path /datadisk/LLaVA-Finetune/llava_v1_5_mix665k.json \
    --image_folder /datadisk/LLaVA-Finetune/images/ \
    --vision_tower /datadisk/llm/clip-vit-large-patch14-336/ \
    --image_aspect_ratio anyres \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_patch_merge_type spatial_unpad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /datadisk/checkpoints/llava-v1.6-13b-lora-sp-pooling \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb

