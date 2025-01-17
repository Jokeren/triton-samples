"""Demonstrate indirect memory access patterns in Triton."""

import torch
import triton
import triton.language as tl


@triton.jit
def get_kernel(x_ptr, index_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
    """Perform indirect memory access using index array."""
    index_offsets = tl.arange(0, BLOCK_SIZE)
    index = tl.load(index_ptr + index_offsets)
    x = tl.load(x_ptr + index[:, None] * BLOCK_SIZE + index[None, :])
    tl.store(y_ptr + index[:, None] * BLOCK_SIZE + index[None, :], x)


# Test parameters
BLOCK_SIZE = 128
index = torch.arange(BLOCK_SIZE, device="cuda", dtype=torch.long)
x = torch.ones((BLOCK_SIZE, BLOCK_SIZE), device="cuda", dtype=torch.long)
y = torch.zeros((BLOCK_SIZE, BLOCK_SIZE), device="cuda", dtype=torch.long)

get_kernel[(1,)](x, index, y, BLOCK_SIZE)
print(y)
