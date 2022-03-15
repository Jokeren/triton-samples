import torch

import triton
import triton.language as tl

@triton.jit
def reshape_kernel(
    x_ptr,
    y_ptr,
    BLOCK_SIZE: tl.constexpr
):
    x_offsets = tl.arange(0, BLOCK_SIZE)[None, :] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None]
    y_offsets = tl.arange(0, BLOCK_SIZE)[None, :] + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE
    # correct
    #x = tl.load(x_ptr + y_offsets)
    #tl.store(y_ptr + x_offsets, x)
    # wrong
    x = tl.load(x_ptr + x_offsets)
    tl.store(y_ptr + y_offsets, x)


BLOCK_SIZE = 128
x = torch.arange((BLOCK_SIZE * BLOCK_SIZE), device='cuda', dtype=torch.long).view((BLOCK_SIZE, BLOCK_SIZE))
y = torch.zeros((BLOCK_SIZE * BLOCK_SIZE), device='cuda', dtype=torch.long).view((BLOCK_SIZE, BLOCK_SIZE))
reshape_kernel[(1,)](x, y, BLOCK_SIZE)

print(y)

