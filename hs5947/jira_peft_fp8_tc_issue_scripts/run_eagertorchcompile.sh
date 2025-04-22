#!/bin/bash
echo "*** TEST EAGER + torch.compile MODE ***"
PT_HPU_LAZY_MODE=0 python quantize_with_lora.py
