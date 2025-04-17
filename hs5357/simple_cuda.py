import torch
import os

def test(device):
    torch.manual_seed(101)
    in_tensor = torch.randn(1, 2, 9, 9).to(device)
    out_tensor = torch.conv2d(in_tensor, torch.randn(2, 2, 8, 8).to(device))
    print(f'out_tensor ({device}): Before Save) - {out_tensor} ')
    torch.save(out_tensor, f'saved_{device}.pkl')
    reloaded_tensor = torch.load(f'saved_{device}.pkl')
    return reloaded_tensor

cpu_out = test('cpu')
print(f"cpu_out - {cpu_out}")
cuda_out = test('cuda')
print(f"cuda_out - {cuda_out.cpu()}")
assert True == torch.equal(cpu_out, cuda_out.cpu())
