import torch

import triton
import triton.language as tl

@triton.jit
def sum_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    STRIDE: tl.constexpr,
    CHANNEL_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    # correct
    #offsets = tl.arange(0, BLOCK_SIZE)[:, None] * STRIDE + tl.arange(0, CHANNEL_SIZE)[None, :]
    #x_val = tl.load(x_ptr + offsets)
    # wrong
    x_val = tl.load(x_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * STRIDE + tl.arange(0, CHANNEL_SIZE)[None, :])
    #######################################
    x_sum = tl.sum(x_val, axis=1)
    tl.store(y_ptr + tl.arange(0, BLOCK_SIZE), x_sum)

BLOCK_SIZE = 128
CHANNEL_SIZE = 5
x = torch.ones((BLOCK_SIZE, BLOCK_SIZE), device='cuda', dtype=torch.long)
y = torch.zeros((BLOCK_SIZE), device='cuda', dtype=torch.long)
z = torch.zeros((BLOCK_SIZE, BLOCK_SIZE), device='cuda', dtype=torch.long)

sum_kernel[(1,)](x, y, z, BLOCK_SIZE, CHANNEL_SIZE, BLOCK_SIZE)
print(y)
