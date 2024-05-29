#!/bin/bash


# Use the online config file by default
IMAGE_PROCESSOR_NAME=${1:-}

# Switch to below to use local preprocessor_config.json by default
# IMAGE_PROCESSOR_NAME=${1:-$(pwd)/preprocessor_config.json}

MODEL=${MODEL:-facebook/detr-resnet-50}
EPOCHS=${EPOCHS:-1}
OUTDIR=${OUTDIR:-/tmp/outputs}
DATASET=${DATASET:-rafaelpadilla/coco2017}
ICNAME=${ICNAME:-image}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-"20"}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-"10"}
USE_CPU=${USE_CPU:-"no"}
START_CLEAN=${START_CLEAN:-"yes"}
WORLD_SIZE=${WORLD_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-2}

[[ ${START_CLEAN} == "yes" ]] && rm -rf ${OUTDIR}

# Works for both Method-A and Method-B reported in GS-123
CMDLINE="run_example.py \
     --model_name_or_path  ${MODEL} \
     --dataset_name ${DATASET} \
     --output_dir ${OUTDIR}  \
     --bf16 false \
     --remove_unused_columns False  --image_column_name ${ICNAME} \
     --do_train --do_eval --learning_rate 2e-4 --num_train_epochs ${EPOCHS} \
     --per_device_train_batch_size ${BATCH_SIZE} --per_device_eval_batch_size ${BATCH_SIZE} \
     --evaluation_strategy epoch  --save_strategy epoch \
     --load_best_model_at_end True \
     --save_total_limit 3  --seed 1337 \
     --dataloader_num_workers 1 \
     --ignore_mismatched_sizes=True \
     --trust_remote_code=True"

[[ ${MAX_TRAIN_SAMPLES} != "" ]] && CMDLINE="${CMDLINE} --max_train_samples ${MAX_TRAIN_SAMPLES}"
[[ ${MAX_EVAL_SAMPLES} != "" ]] && CMDLINE="${CMDLINE} --max_eval_samples ${MAX_EVAL_SAMPLES}"
[[ ${USE_CPU} == "no" ]] && CMDLINE="${CMDLINE} --use_habana  --use_lazy_mode --use_hpu_graphs_for_inference --gaudi_config_name Habana/vit  --throughput_warmup_steps 3"
[[ ${IMAGE_PROCESSOR_NAME} != "" ]] && CMDLINE="${CMDLINE} --image_processor_name=${IMAGE_PROCESSOR_NAME}"

if [[ ${WORLD_SIZE} -eq 1 ]]
then
    MP_CMDLINE="python ${CMDLINE}"
else
    MP_CMDLINE="python ../gaudi_spawn.py --world_size ${WORLD_SIZE} --use_mpi ${CMDLINE}"
fi

echo  ${MP_CMDLINE} 2>&1 | tee cmdline.log
time ${MP_CMDLINE} 2>&1 | tee result.log
