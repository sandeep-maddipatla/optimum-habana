import torch
import random
import numpy as np

 ## @torch.compiler.disable
class SimpleTest:
    def __init__(self, index):
        self.step_index = index if (index >= 0) else None
        self.sigmas = torch.rand(10, device='hpu')
 
    @torch.compile(backend='hpu_backend')
    def step(self):
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        retval = sigma_next - sigma

        # upon completion increase step index by one
        self.step_index += 1

        return retval

torch.set_num_threads(1)
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

idx = 4
st = SimpleTest(idx)
x = st.step()
y = st.step()
print(x)
print(y)