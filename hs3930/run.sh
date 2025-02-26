# cpu
python3 customer_reproducer.py
# hpu lazy
DEVICE="hpu" PT_HPU_LAZY_MODE=1 python3 customer_reproducer.py
# hpu eager
DEVICE="hpu" PT_HPU_LAZY_MODE=0 python3 customer_reproducer.py
python3 plot_curves.py
