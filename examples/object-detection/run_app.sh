#!/bin/bash


# Use the online config file by default
IMAGE_PROCESSOR_NAME=${1:-}

# Switch to below to use local preprocessor_config.json by default
# IMAGE_PROCESSOR_NAME=${1:-$(pwd)/preprocessor_config.json}

MODEL=${MODEL:-facebook/detr-resnet-50}
EPOCHS=${EPOCHS:-1}
OUTDIR=${OUTDIR:-/tmp/outputs}
DATASET=${DATASET:-cppe-5}
MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES:-""}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-""}
USE_CPU=${USE_CPU:-"no"}
START_CLEAN=${START_CLEAN:-"yes"}
WORLD_SIZE=${WORLD_SIZE:-1}
BATCH_SIZE=${BATCH_SIZE:-8}

# We attempt to use a Gaudi config with all of below fields set to true
# The Habana/vit config allows us to do this. So we specify this as the config.
# More Details: https://huggingface.co/docs/optimum/en/habana/package_reference/gaudi_config
# {
#    "use_fused_adam": true,
#    "use_fused_clip_norm": true,
#    "use_torch_autocast": true
# }
GAUDI_CONFIG_NAME=Habana/vit

[[ ${START_CLEAN} == "yes" ]] && rm -rf ${OUTDIR}

CMDLINE="run_object_detection.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --do_train false \
    --do_eval true \
    --output_dir ${OUTDIR} \
    --num_train_epochs ${EPOCHS} \
    --image_square_size 600 \
    --learning_rate 5e-5 \
    --weight_decay 1e-4 \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps 1 \
    --remove_unused_columns false \
    --ignore_mismatched_sizes true \
    --metric_for_best_model eval_map \
    --greater_is_better true \
    --load_best_model_at_end true \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 3 \
    --seed 1337 \
    --eval_do_concat_batches false"


[[ ${MAX_TRAIN_SAMPLES} != "" ]] && CMDLINE="${CMDLINE} --max_train_samples ${MAX_TRAIN_SAMPLES}"
[[ ${MAX_EVAL_SAMPLES} != "" ]] && CMDLINE="${CMDLINE} --max_eval_samples ${MAX_EVAL_SAMPLES}"
[[ ${USE_CPU} == "no" ]] && CMDLINE="${CMDLINE} --use_habana  --use_lazy_mode --gaudi_config_name ${GAUDI_CONFIG_NAME}  --throughput_warmup_steps 3"
[[ ${IMAGE_PROCESSOR_NAME} != "" ]] && CMDLINE="${CMDLINE} --image_processor_name=${IMAGE_PROCESSOR_NAME}"

if [[ ${WORLD_SIZE} -eq 1 ]]
then
    MP_CMDLINE="python ${CMDLINE}"
else
    MP_CMDLINE="python ../gaudi_spawn.py --world_size ${WORLD_SIZE} --use_mpi ${CMDLINE}"
fi

echo  ${MP_CMDLINE} 2>&1 | tee cmdline.log
time ${MP_CMDLINE} 2>&1 | tee result.log
