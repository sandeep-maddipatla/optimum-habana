#!/bin/bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export INSTANCE_DIR="dog"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="out"
export PT_HPU_LAZY_MODE=${PT_HPU_LAZY_MODE:-0}
export TRAIN_STEPS=${TRAIN_STEPS:-100}

if [ "${PT_TOWL_LOG_ENABLE}" == "1" ]; then
  echo "PT_TOWL_LOG_ENABLE=${PT_TOWL_LOG_ENABLE}. Including additional env variables"
  export PT_HPU_ENABLE_EXECUTION_THREAD_NO_WAIT=1
  export PT_HPU_EAGER_4_STAGE_PIPELINE_ENABLE=0
  export PT_HPU_EAGER_PIPELINE_ENABLE=0
  export PT_HPU_DISABLE_ASYNC_COLLECTIVE=1
  export PT_HPU_ENABLE_LAZY_EAGER_LAUNCH_EXEC_THREAD=0
  export PT_HPU_ENABLE_LAZY_EAGER_EXECUTION_THREAD=0
  export PT_HPU_ENABLE_COMPILE_THREAD=0
  export PT_HPU_ENABLE_EXECUTION_THREAD=0
  export PT_HPU_LAZY_ACC_PAR_MODE=0
  export PT_HPU_SYNC_LAUNCH=1
  env
fi

python train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$CLASS_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --num_class_images=200 \
  --mixed_precision=bf16 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=$TRAIN_STEPS \
  --gaudi_config_name Habana/stable-diffusion \
 boft

 
