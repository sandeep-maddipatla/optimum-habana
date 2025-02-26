import torch
import torch.nn as nn
import habana_frameworks.torch.core as htcore
import os

class MyTanh:
    def __init__(self, x):
        self.f = nn.Tanh()
        self.x = x
    def get_tanh(self, x):
        return self.f(x)
    def backward(self):
        y = self.f(x)
        return y.backward()
    
torch.set_printoptions(precision=40)
# Run on HPU
x = torch.tensor(0.3547980487346649, requires_grad=True, dtype=torch.float32).to("hpu")
# retain_grad() to avoid error on x.grad for non leaf tensors
x.retain_grad()
c = MyTanh(x)
if os.getenv("PT_HPU_LAZY_MODE", "1") == "0":
    # Needs to be a callable function and should be able to run backward, hence class MyTanh
    c.get_tanh = torch.compile(c.get_tanh, backend="hpu_backend")
c.backward() 
print("x.grad(hpu):", c.x.grad.cpu()) 

# Repeat on CPU
x_cpu = torch.tensor(0.3547980487346649, requires_grad=True, dtype=torch.float32).to("cpu")
x_cpu.retain_grad()
f = nn.Tanh()
y_cpu = f(x_cpu)  
y_cpu.backward()
print("x_cpu.grad(cpu):", x_cpu.grad.cpu()) 