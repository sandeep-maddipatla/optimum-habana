#!/bin/bash
echo "*** TEST PURE EAGER MODE ***"
PT_HPU_LAZY_MODE=0 MODE=pure-eager python quantize_with_lora.py
