import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from habana_frameworks.torch.hpex.optimizers import FusedSGD

LR = 0.6
RHO = 0.1
WEIGHT_DECAY = 0.0
MOMENTUM = 0.0


SEED = 0


import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.random as htrandom
from SAM import SAM
from habana_frameworks.torch.hpex.optimizers import FusedSGD

htrandom.manual_seed_all(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

device = torch.device(os.getenv('DEVICE','cpu'))
    
class TanhFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.nn.functional.tanh(input)
        return output    
    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        tanh_grad = 1.0 - torch.nn.functional.tanh(input) ** 2  # Derivative of tanh
        return grad_output * tanh_grad
    

class MinimalModel(nn.Module):
    linear1: nn.Module
    linear2: nn.Module

    def __init__(self):
        super(MinimalModel, self).__init__()
        self.linear1: nn.Module = nn.Linear(in_features=2, out_features=1, bias=False)
        self.linear2: nn.Module = nn.Linear(in_features=1, out_features=1, bias=False)
    
    def forward(self, x):
        out1 = self.linear1(x)
        out1 = TanhFunction.apply(out1)
        out2 = self.linear2(out1)
        out2 = out2.flatten()
        return out2

model = MinimalModel().to(device)
if os.getenv('PT_HPU_LAZY_MODE') != '1' and os.getenv('DEVICE')=='hpu':
    model = torch.compile(model, backend="hpu_backend")

loss_func = nn.MSELoss()

X = np.array([[1.05, 0.95], [0.95, 1.05]])
y = np.array([1.0, -1.0])
epoch_nums = 1000

X = torch.tensor(X).to(device, dtype=torch.float32)
y = torch.tensor(y).to(device, dtype=torch.float32)

optimizer = SAM(model.parameters(), optim.SGD, lr=LR, rho=RHO, 
                weight_decay=WEIGHT_DECAY, momentum=MOMENTUM, is_lazy=os.getenv('PT_HPU_LAZY_MODE') == '1')



loss_list = []
state_dict_list = []
pbar = tqdm(range(epoch_nums))

# Full batch SAM
for epoch in pbar:
    optimizer.zero_grad()
    out = model(X)
    loss = loss_func(out, y)
    loss_list.append(loss.item())
    pbar.set_postfix({'loss': loss.item()})
    detached_state_dict = {}
    for name, param in model.state_dict().items():
        detached_state_dict[name] = param.detach().cpu().numpy()
    state_dict_list.append(detached_state_dict)
    loss.backward()
    if os.getenv('PT_HPU_LAZY_MODE') == '1':
        htcore.mark_step()

    optimizer.first_step(zero_grad=True) # First Step
    if os.getenv('PT_HPU_LAZY_MODE') == '1':
        htcore.mark_step()

    second_loss = loss_func(model(X), y)
    second_loss.backward()
    if os.getenv('PT_HPU_LAZY_MODE') == '1':
        htcore.mark_step()
    optimizer.second_step(zero_grad=True) # Second Step
    if os.getenv('PT_HPU_LAZY_MODE') == '1':
        htcore.mark_step()

import pickle as pkl
to_save = [loss_list, state_dict_list]
suffix = os.getenv('SUFFIX','')
mode = "cpu"
if device.type == "hpu":
    if os.getenv('PT_HPU_LAZY_MODE') == '1':
        mode = "lazy"    
    else:
        mode = "eager" 
with open(f"verify_{mode}_{suffix}.pkl", 'wb') as f:
    pkl.dump(to_save, f, protocol=pkl.HIGHEST_PROTOCOL)
