import torch
import random
from habana_frameworks.torch.hpex.normalization import FusedRMSNorm

def RmsNormTCFunction(test_tensor, weight, eps=1e-06):
    result = FusedRMSNorm.apply(test_tensor, weight, eps, use_stages=True, bwd_mode=0, fast_math=False)
    return result

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
random.seed(0)

weight = torch.rand([2,3], dtype=torch.float).to('hpu')
test_tensor = torch.rand([2,3], dtype=torch.float).to('hpu')

print(f'weight = {weight}')
print(f'test_tensor = {test_tensor}')

print("FusedRMSNorm default path")
RmsNormTCF = torch.compile(RmsNormTCFunction, backend="hpu_backend", dynamic=True)
result = RmsNormTCF(test_tensor, weight)
print(result)