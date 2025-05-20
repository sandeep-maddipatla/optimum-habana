import torch

def foo(a):
    tmp0 = a.shape[1]//8
    a_subset = a[:, tmp0]
    tmp1 = tmp0 ** 0.5
    tmp2 = 1/tmp1
    scale = tmp2 ** 0.5
    b = a_subset * scale
    return b

fn = torch.compile(foo, backend='hpu_backend')
device = 'hpu'
a = torch.rand([2, 8]).to(device)
x = fn(a)
print(f'{x=}')
p = torch.rand([2,16]).to(device)
y = fn(p)
print(f'{y=}')


