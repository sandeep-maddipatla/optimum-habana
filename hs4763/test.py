import torch
import habana_frameworks.torch.core as htcore

a = torch.randn(3,3, device='hpu')
print(f'a = {a}')
x = torch.is_complex(a)
print(f'torch.is_complex(a) = {x}')
b = torch.linalg.eigvals(a)
print(f'torch.linalg.eigvals(a) = {b}')
