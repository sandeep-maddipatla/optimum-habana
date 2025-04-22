#!/bin/bash
git clone https://github.com/huggingface/optimum-habana.git
cd optimum-habana
pip install -e .
cd ..

git clone https://github.com/dsocek/peft.git -b add_inc_dispatcher
cd peft
pip install -e .
cd ..

pip install sentencepiece

huggingface-cli login

echo "======================== INC - Measure Stats for FP8 Quantization ======================"
cp -r optimum-habana/examples/stable-diffusion/quantization .
PT_HPU_LAZY_MODE=0 /usr/bin/env python measure_stats.py
