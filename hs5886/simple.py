import torch
import random
import numpy as np

 ## @torch.compiler.disable
class SimpleTest:
    def __init__(self, index):
        self.step_index = index if (index >= 0) else None
        self.sigmas = torch.rand(10, device='hpu')
 
    @torch.compile(backend='hpu_backend')
    def step(self, sample, model_output):
   
        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self.step_index += 1

        return (prev_sample,)

torch.set_num_threads(1)
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

idx = 4
sample = torch.rand(1, device='hpu')
model_out = torch.rand(1, device='hpu')

st = SimpleTest(4)
x = st.step(sample, model_out)
y = st.step(sample, model_out)
print(x)
print(y)