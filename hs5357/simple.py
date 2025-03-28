import torch
import habana_frameworks.torch.core as htcore

input = torch.zeros(2, 3).to(dtype=torch.float32, device='hpu')
input[0] = 0
input[1] = 1
print(f'input = {input}')

permuted_input = input.permute(1,0)
print(f'permuted_input = {permuted_input}')

storage = permuted_input.untyped_storage()
output = storage.cpu()
print(f'output = {output}')


