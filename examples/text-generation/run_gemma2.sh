#!/bin/bash
python run_generation.py \
--model_name_or_path google/gemma-2-9b-it \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens 1024 \
--bf16 \
--attn_softmax_bf16 \
--do_sample \
--prompt_jskey ${PROMPT:-LLAMA_2048}

##prompt ${PROMPT:-"What is the capital of France"}