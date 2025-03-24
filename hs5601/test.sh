#!/bin/bash
set -x
localdir=$(pwd)
logdir=${1:-${localdir}}
logfile=$(echo $logdir)_logfile.log
workdir=${HOME}/optimum-habana/examples/image-classification

cd ${workdir}
pip install -r requirements.txt
pip install optimum-habana
pip list | grep habana
python run_timm_example.py \
  --model_name_or_path google/vit-base-patch16-224-in21k \
  --dataset_name cifar10 \
  --do_train --do_eval \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --per_device_train_batch_size 128 \
  --per_device_eval_batch_size 64 \
  --optim adamw_torch \
  --output_dir /tmp/outputs/ \
  --bf16 \
  --torch_compile True \
  --use_habana \
  --use_hpu_graphs \
  --use_lazy_mode False \
  --throughput_warmup_steps 6 \
  --gaudi_config_name Habana/vit 2>&1 | tee ${logfile}

cd ${localdir}