import torch
import random
import os
import numpy as np

def get_device():
    return os.getenv("DEVICE", "hpu")

def get_compile_backend():
    device = get_device()
    compile_backend = 'hpu_backend' if device=='hpu' else 'inductor'
    print(f'torch.compile config: device = {device} .. compile_backend = {compile_backend}')
    return compile_backend

 ## @torch.compiler.disable
class SimpleTest:
    def __init__(self, index):
        self.step_index = index if (index >= 0) else None
        self.device = get_device()
        self.sigmas = torch.rand(10, device=self.device)
 
    @torch.compile(backend=get_compile_backend())
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
print(f'First step call: x={x}')
y = st.step()
print(f'Second step call: y={y}')
