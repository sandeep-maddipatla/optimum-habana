#!/bin/bash

HABANA_LOGS=${HABANA_LOGS:-$(pwd)/habana_logs}
HL_SMI_CMD="hl-smi -l 1 -f csv -Q timestamp,name,driver_version,temperature.aip,utilization.aip,memory.total,memory.free,memory.used"
HL_SMI_LOG="hlsmi.log"

# Clean start
OUTPUT_DIR=${OUTPUT_DIR:-output}
OUTPUT_DIR=${OUTPUT_DIR}_$(date +%h_%d_%H_%M_%S)
rm -r ${OUTPUT_DIR}
rm -f *.log *.json *.gz *.hltv
mkdir -p ${OUTPUT_DIR}

pkill hl-smi 2>/dev/null
rm -f ${HL_SMI_LOG} 2>/dev/null0
rm -f /root/metricslog.json

START_TIME=$(date)q
${HL_SMI_CMD} 2>&1 > ${HL_SMI_LOG} &lsfid

python run_generation.py \
--model_name_or_path ${MODEL:-google/gemma-2-9b-it} \
--use_hpu_graphs \
--use_kv_cache \
--max_new_tokens ${MAX_TOKENS:-1024} \
--bf16 \
--attn_softmax_bf16 \
--do_sample \
--prompt_jskey ${PROMPT:-LLAMA_2048} \
--profiling_warmup_steps ${PROFILING_WARMUP_STEPS:-10} \
--profiling_steps ${PROFILING_STEPS:-40}

##prompt ${PROMPT:-"What is the capital of France"}

pkill hl-smi

python3 script/throughput.py summary.json 2>&1 | tee summary.log

# Gather outputs list
env > env.log
pip list > pip_list.log
mv *.log *.json ${OUTPUT_DIR}
mv *.json.gz ${OUTPUT_DIR} 2>/dev/null
mv *.hltv ${OUTPUT_DIR} 2>/dev/null
mv /root/metricslog.json ${OUTPUT_DIR} 2>/dev/nullq
cp -r $(pwd)/script ${OUTPUT_DIR} 2>/dev/null
mv ${HL_SMI_LOG} ${OUTPUT_DIR} 2>/dev/null

END_TIME=$(date)
echo START ${START_TIME} | tee ${OUTPUT_DIR}/time.log
echo STOP ${END_TIME} | tee -a ${OUTPUT_DIR}/time.log

chmod -R 777 ${OUTPUT_DIR} && echo Saved logs in ${OUTPUT_DIR}
echo Exiting with code