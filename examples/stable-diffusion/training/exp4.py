import torch
import sys
from contextlib import nullcontext

def cayley_batch(data):
    """
    Perform the Cayley parametrization on a batch of skew-symmetric matrices.

    Args:
        data: A batch of skew-symmetric matrices of shape (b, r, c).
    """
    b, r, c = data.shape
    # Ensure the input matrix is skew-symmetric
    skew_mat = 0.5 * (data - data.transpose(1, 2))
    id_mat = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

    # Perform the Cayley parametrization
    Q = torch.linalg.solve(id_mat + skew_mat, id_mat - skew_mat, left=False)

    return Q

device = 'hpu'
fn = torch.compile(cayley_batch, backend='hpu_backend')

in_shape1 = [360, 8, 8]

in_tensor1 = torch.randn(in_shape1, requires_grad=True, device=device)

y = fn(in_tensor1)
z = y.sum()
z.backward()
print(f'{y.shape=}')
print(f'{in_tensor1.grad.shape}')

s1_list = [160, 160, 360, 40, 96, 
           720, 1440, 2880, 2160, 1080, 
           1440, 2880, 2160, 1080,  1440, 
           2880, 2160, 1080, 80, 96]

enable_profile = False
with nullcontext():
    for i in range(1):
        for s1 in s1_list:
            in_shape2 = [s1, 8, 8]
            print(f'{in_shape2=}')
            in_tensor2 = torch.randn(in_shape2, requires_grad=True, device=device)
            y = fn(in_tensor2)
            z = y.sum()
            z.backward()
            print(f'{in_tensor2.grad.shape=}')