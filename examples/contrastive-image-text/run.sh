#!/bin/bash
set -x
cd $HOME/optimum-habana
pip install -e .
cd $HOME/optimum-habana/examples/contrastive-image-text
pip install -r requirements.txt
pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.19.0
echo python3 ../gaudi_spawn.py --world_size 1 --use_deepspeed run_clip.py     --output_dir=/tmp/clip_roberta     --model_name_or_path=./clip-roberta     --data_dir /mnt/weka/data/pytorch/coco2017/     --dataset_name ydshieh/coco_dataset_script     --dataset_config_name 2017     --image_column image_path     --caption_column caption     --remove_unused_columns=False     --do_train --do_eval     --mediapipe_dataloader     --per_device_train_batch_size="64"     --per_device_eval_batch_size="64"     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1     --overwrite_output_dir     --use_habana        --gaudi_config_name="Habana/clip"     --throughput_warmup_steps=30     --save_strategy="no"     --dataloader_num_workers=2     --use_hpu_graphs     --max_steps=100       --logging_nan_inf_filter     --trust_remote_code  --deepspeed ./ds_config.json | tee result.log
python3 ../gaudi_spawn.py --world_size 1 --use_deepspeed run_clip.py     --output_dir=/tmp/clip_roberta     --model_name_or_path=./clip-roberta     --data_dir /mnt/weka/data/pytorch/coco2017/     --dataset_name ydshieh/coco_dataset_script     --dataset_config_name 2017     --image_column image_path     --caption_column caption     --remove_unused_columns=False     --do_train --do_eval     --mediapipe_dataloader     --per_device_train_batch_size="64"     --per_device_eval_batch_size="64"     --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1     --overwrite_output_dir     --use_habana        --gaudi_config_name="Habana/clip"     --throughput_warmup_steps=30     --save_strategy="no"     --dataloader_num_workers=2     --use_hpu_graphs     --max_steps=100       --logging_nan_inf_filter     --trust_remote_code  --deepspeed ./ds_config.json 2>&1 | tee -a result.log

