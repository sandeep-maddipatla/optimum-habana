#!/bin/bash
python run_generation.py \
--model_name_or_path google/gemma-2-9b-it \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 256 \
--batch_size 1 \
--bf16 \
--attn_softmax_bf16 \
--do_sample \
--prompt "What is the capital of france"