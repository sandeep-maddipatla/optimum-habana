#!/bin/bash
set -x
localdir=$(pwd)
logfile=${1:-${logdir}/hs5601_logfile.log}
workdir=${HOME}/optimum-habana/examples/image-classification

cd ${workdir}
pip install -r requirements.txt
pip install optimum-habana
pip list | grep habana
PT_HPU_LAZY_MODE=0 python run_image_classification.py \
  --model_name_or_path google/vit-base-patch16-224-in21k \
  --dataset_name cifar10 \
  --image_column_name img \
  --remove_unused_columns False \
  --overwrite_output_dir \
  --do_train --do_eval \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 64 \
  --optim adamw_torch \
  --output_dir /tmp/outputs/ \
  --bf16 \
  --torch_compile True \
  --use_lazy_mode False \
  --torch_compile_backend hpu_backend \
  --use_habana \
  --throughput_warmup_steps 6 \
  --gaudi_config_name Habana/vit 2>&1 | tee ${logfile}

cd ${localdir}
chmod 777 ${logfile}