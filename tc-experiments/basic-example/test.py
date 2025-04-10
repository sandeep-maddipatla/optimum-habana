import torch
import torch.nn as nn
import torch.optim as optim
import habana_frameworks.torch.core as htcore
import habana_frameworks.torch.hpu.random as htrandom
import argparse
import os
import random
import numpy as np

# Define a simple model with only a tanh activation layer
class TanhModel(nn.Module):
    def __init__(self):
        super(TanhModel, self).__init__()
        self.linear = nn.Linear(4, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        y = self.linear(x)
        y = self.activation(y)
        return y

def is_lazy_mode():
    return os.getenv('PT_HPU_LAZY_MODE') == '1'

def mark_step(device):
    if device == 'hpu' and is_lazy_mode():
        htcore.mark_step()

def run(device='hpu', skip_torch_compile=False):
    # Create an instance of the model
    model = TanhModel()
    model = model.to(device)
    if device=='hpu' and not is_lazy_mode() and not skip_torch_compile:
        print('Running torch.compile with hpu_backend')
        model = torch.compile(model, backend='hpu_backend')
        print(f'model = {model}')

    # Hyper-parameters
    learning_rate = 0.01
    loss_func = nn.MSELoss()

    # Example input tensor
    example_input = torch.tensor([1.0, -1.0, 0.0, 0.5]).to(device)
    target = torch.tensor([0.0]).to(device)

    # Forward pass through the model
    output = model(example_input)
    print(f'output = {output}')

    # Backward pass
    loss = loss_func(output, target)
    print(f'loss = {loss}')
    loss.backward()
    mark_step(device)
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Gradient of {name}: {param.grad}")

    

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