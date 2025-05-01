import torch
import os

def get_device():
    return os.getenv("DEVICE", "hpu")

def get_compile_backend():
    device = get_device()
    default_compile_backend = 'hpu_backend' if device=='hpu' else 'inductor'
    backend = os.getenv("BACKEND", "default")
    compile_backend = default_compile_backend if backend == 'default' else backend
    print(f'torch.compile config: device = {device} .. compile_backend = {compile_backend}')
    return compile_backend

 ## @torch.compiler.disable
class SimpleTest:
    def __init__(self, index):
        self.step_index = index if (index >= 0) else None
        self.device = get_device()
        self.sigmas = torch.arange(0, 16, device=self.device)
 
    @torch.compile(backend=get_compile_backend())
    def step(self):
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        retval = sigma_next + sigma

        # upon completion increase step index by one
        self.step_index += 1

        return retval

idx = 4
st = SimpleTest(idx)
x = st.step()
assert x == 2*idx + 1
print(f'First step call: x={x}')
y = st.step()
assert y == 2*idx + 3
print(f'Second step call: y={y}')
