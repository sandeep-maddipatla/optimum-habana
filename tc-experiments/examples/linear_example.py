import torch
import torch.nn as nn
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.random as htrandom
import argparse
import os
import random
import numpy as np

# Define a simple model with only a linear layer
class LinearModel(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.linear = torch.nn.Linear(4, 3) 

    def forward(self, x): 
        return self.linear(x)

def run(device='hpu', skip_torch_compile=False):
    # Create an instance of the model
    model = LinearModel()
    model = model.to(device)
    if os.getenv('PT_HPU_LAZY_MODE') != '1' and device=='hpu' and not skip_torch_compile:
        print('Running torch.compile with hpu_backend')
        model = torch.compile(model, backend='hpu_backend')
        print(f'model = {model}')

    # Example input tensor
    x = torch.tensor([1.0, -1.0, 0.0, 0.5], device=device, requires_grad=True)
    print(f'x = {x}')

    # Forward pass through the model
    y = model(x)
    print(f'y = {y}')

    # Trigger implicit gradient computation
    y.sum().backward()
    print(f'gradient dy/dx = {x.grad}')

def get_args():
    parser = argparse.ArgumentParser(description='Args for run script')
    parser.add_argument('--device', type=str, choices=['cpu', 'hpu'], default='hpu', help='Device to run the model on. [cpu/hpu]')
    parser.add_argument('--pure-eager', action='store_true', help='Skip torch.compile in eager mode')
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()

    # set seeds
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    state = htrandom.get_rng_state()
    htrandom.set_rng_state(state)
    initial_seed = htrandom.initial_seed()
    htrandom.manual_seed(seed)
    run(device=args.device, skip_torch_compile=args.pure_eager)