import torch

import triton
import triton.language as tl

@triton.jit
def add_kernel(
    y_ptr,
    n, t, m, P, BLOCK_SIZE: tl.constexpr
):
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(y_ptr + offsets, (t * m) % P)

BLOCK_SIZE = 128
P = 2038074743
t = 1023
m = 4096 * 4096
y = torch.zeros((BLOCK_SIZE,), device='cuda', dtype=torch.long)
print('Python: {} % {} = {}'.format(t * m, P, (t * m) % P))
add_kernel[(1,)](y, BLOCK_SIZE, t, m, P, BLOCK_SIZE)
print('Triton: {}'.format(y[0].item()))

t = 3
print('Python: {} % {} = {}'.format(t * m, P, (t * m) % P))
add_kernel[(1,)](y, BLOCK_SIZE, t, m, P, BLOCK_SIZE)
print('Triton: {}'.format(y[0].item()))
