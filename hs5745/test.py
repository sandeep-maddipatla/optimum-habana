#### test.py
# Run with below command
# PT_HPU_LAZY_MODE=0 python test.py

import torch
import habana_frameworks.torch as ht
a = torch.randn(1,2,3,4)
b = a.to(device='hpu', dtype=torch.bfloat16)
print(f'b.is_contiguous() = {b.is_contiguous()}')
c = b.to(memory_format=torch.channels_last)
print(f'c.is_contiguous() = {c.is_contiguous()}')
d = a.to(device='hpu', dtype=torch.bfloat16)
d = d.to(memory_format=torch.channels_last)
e = b.to('hpu')
print(d)
print(f'd.is_contiguous() = {d.is_contiguous()}')