import torch

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 20)

    def forward(self, x):
        return self.linear(x).relu()

model = MyModule()

model = model.to(device="hpu")
model = torch.compile(model, backend="hpu_backend")

inp = torch.rand(10,10).to(device="hpu")
out = model(inp)
print(f'{out=}')
