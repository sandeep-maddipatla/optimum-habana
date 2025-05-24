import torch

def foo(a, n, hxw):
    b = a.view(n, -1, hxw).transpose(1,2)
    return b


device = 'hpu'
fn = torch.compile(foo, backend='hpu_backend')

n, c, h, w = (1, 2, 3, 4)
x = torch.randn([n,c,h,w], requires_grad=True, device=device)
y = fn(x, n, h*w).sum()
y.backward()
print(f'{x.grad=}')
print(f'{x.shape=} ... {y.shape=}')

n, c, h, w = (1, 2, 5, 4)
x = torch.randn([n,c,h,w], requires_grad=True, device=device)
y = fn(x, n, h*w).sum()
y.backward()
print(f'{x.grad=}')
print(f'{x.shape=} ... {y.shape=}')
