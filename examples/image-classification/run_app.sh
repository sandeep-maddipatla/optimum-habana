#!/bin/bash
IMAGE_PROCESSOR_NAME=${1:-https://huggingface.co/timm/resnetv2_50x1_bit.goog_in21k/raw/main/config.json}

# Corresponds to Method-A reported in GS-123
CMDLINE="python run_image_classification.py \
     --model_name_or_path  timm/resnetv2_50x1_bit.goog_in21k \
     --dataset_name cifar10 \
     --output_dir /tmp/outputs/  \
     --remove_unused_columns False  --image_column_name img \
     --do_train --do_eval --learning_rate 2e-4 --num_train_epochs 5 \
     --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \
     --evaluation_strategy epoch  --save_strategy epoch \
     --load_best_model_at_end True \
     --save_total_limit 3  --seed 1337 \
     --use_habana  --use_lazy_mode --use_hpu_graphs_for_inference --gaudi_config_name Habana/vit  --throughput_warmup_steps 3 \
     --dataloader_num_workers 1 \
     --ignore_mismatched_sizes=True  --trust_remote_code=True \
     --use_safetensors False \
     --image_processor_name=${IMAGE_PROCESSOR_NAME}"

echo ${CMDLINE}
time ${CMDLINE} 2>&1 | tee result.log
