import torch
import random
from habana_frameworks.torch.hpex.normalization import FusedRMSNorm

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
random.seed(0)

HPUFusedRMSNorm = FusedRMSNorm

weight = torch.rand([2,3], dtype=torch.float).to('hpu')
test_tensor = torch.rand([2,3], dtype=torch.float).to('hpu')

print(f'weight = {weight}')
print(f'test_tensor = {test_tensor}')

print("FusedRMSNorm fast path")
result = HPUFusedRMSNorm.apply(test_tensor, weight, 1e-06, True, 0, True)
print(result)


