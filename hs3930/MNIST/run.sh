# cpu
python3 mnist_sample.py
# hpu lazy
PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=1 python3 mnist_sample.py
# hpu eager
PT_HPU_GPU_MIGRATION=1 PT_HPU_LAZY_MODE=0 python3 mnist_sample.py --use-torch-compile
