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

grads = {}
record_mode =  "lazy" if os.getenv("PT_HPU_LAZY_MODE","1") =="1" else "eager"
bwd_epoch_counter=0
torch.set_printoptions(precision=10)


def dump_tensor(epoch, tensor, tensor_name, stage):
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor)
        
    cpu_tensor = tensor.to("cpu").detach()
    if not os.path.exists(f"{record_mode}/fwd/{epoch}"):
        os.makedirs(f"{record_mode}/fwd/{epoch}")
    torch.save(cpu_tensor, f'{record_mode}/fwd/{epoch}/{stage}_{tensor_name}.pt')
    
def store(grad, parent):
    grads[parent] = grad.clone()

def dump_bwd_tensor(epoch):
    if not os.path.exists(f"{record_mode}/bwd/{epoch}"):
        os.makedirs(f"{record_mode}/bwd/{epoch}")
    for k,v in grads.items():
        v = v.to("cpu").detach()
        torch.save(v, f'{record_mode}/bwd/{epoch}/{k}.pt')

    
class MinimalModel(nn.Module):
    linear1: nn.Module
    linear2: nn.Module

    def __init__(self):
        super(MinimalModel, self).__init__()
        self.linear1: nn.Module = nn.Linear(in_features=2, out_features=1, bias=False)
        self.linear2: nn.Module = nn.Linear(in_features=1, out_features=1, bias=False)
        self.activation = nn.Tanh()
        self.epoch_counter = 0
    
    def forward(self, x):
        a = self.linear1(x)
        a.register_hook(lambda grad:store(grad,'a_lin1'))
        dump_tensor(self.epoch_counter, a, "a", "linear1")
        b = self.activation(a)
        b.register_hook(lambda grad:store(grad,'b_act'))
        dump_tensor(self.epoch_counter, b, "b", "activation")
        c = self.linear2(b)
        c.register_hook(lambda grad:store(grad,'c_lin2'))
        dump_tensor(self.epoch_counter, c, "c", "linear2")
        d = c.flatten()
        d.register_hook(lambda grad:store(grad,'d_flat'))
        dump_tensor(self.epoch_counter, d, "d", "flatten")
        self.epoch_counter+=1

        return d


model = MinimalModel().to(device)
if os.getenv('PT_HPU_LAZY_MODE') != '1' and os.getenv('DEVICE')=='hpu':
    model = torch.compile(model, backend="hpu_backend")

loss_func = nn.MSELoss()

X = np.array([[1.05, 0.95], [0.95, 1.05]])
y = np.array([1.0, -1.0])
epoch_nums = 200

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
    dump_tensor(bwd_epoch_counter, loss.item(), "loss", "loss")
    detached_state_dict = {}
    for name, param in model.state_dict().items():
        detached_state_dict[name] = param.detach().cpu().numpy()
    state_dict_list.append(detached_state_dict)
    loss.backward()

    dump_bwd_tensor(bwd_epoch_counter)
    bwd_epoch_counter+=1

    if os.getenv('PT_HPU_LAZY_MODE') == '1':
        htcore.mark_step()

    optimizer.first_step(zero_grad=True) # First Step
    if os.getenv('PT_HPU_LAZY_MODE') == '1':
        htcore.mark_step()

    second_loss = loss_func(model(X), y)
    second_loss.backward()
    dump_bwd_tensor(bwd_epoch_counter)
    bwd_epoch_counter+=1

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
