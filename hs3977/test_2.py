import torch
import random
from habana_frameworks.torch.hpex.normalization import FusedRMSNorm

torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
random.seed(0)

HPUFusedRMSNorm = FusedRMSNorm
input_shape = [2,4,3]
weight = torch.ones(input_shape, dtype=torch.float).to('hpu')
weight[0, 0,:] = 1
weight[0, 1,:] = 10
weight[0, 2, :] = 2
weight[0, 3, :] = 3
test_tensor = torch.rand(input_shape, dtype=torch.float).to('hpu')

print(f'weight = {weight}')
print(f'test_tensor = {test_tensor}')

print("FusedRMSNorm fast path")
result = HPUFusedRMSNorm.apply(test_tensor, weight, 1e-06, True, 0, True)
print(result)

print("FusedRMSNorm default path")
result = HPUFusedRMSNorm.apply(test_tensor, weight, 1e-06, True, 0, False)
print(result)