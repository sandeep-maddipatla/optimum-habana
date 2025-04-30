import torch
import habana_frameworks.torch.hpu as hthpu

class Model2(torch.nn.Module):
    def __init__(self, inp_size, out_size, inner_size):
        super().__init__()
        self.l1 = torch.nn.Linear(inp_size, inner_size)
        self.l2 = torch.nn.Linear(inner_size, out_size)
        self.tmp = None

    def forward(self, x):
        res = self.l1(x)
        self.tmp = res
        out = self.l2(res)
        return out

torch.set_num_threads(1)
seed = 42
random.seed(seed)
torch.manual_seed(seed)

device = torch.device('hpu')
BS, inp_size, out_size, inner_size = 1, 50000, 20000, 35000

model = Model2(inp_size, out_size, inner_size).to(device)
compiled_model = torch.compile(model, backend='hpu_backend')

inp = torch.randn(BS, inp_size, device=device)
out = compiled_model(inp)

print(f'\n\n1st assignment -> {compiled_model.tmp.sum().item()=}\n\n')
