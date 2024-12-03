import torch
  
import triton
import triton.language as tl

@triton.jit
def permute2d_col(
    x,
    index,
    SIZE,
):
    indicator = tl.arange(0, SIZE)[:, None] == index
    return tl.sum(indicator[None, :] * x[:, :, None], axis=1)


@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    BLOCK_SIZE: tl.constexpr
):
    permute_tid = (tl.arange(0, BLOCK_SIZE) + 8) % BLOCK_SIZE
    tid = tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    x = tl.load(x_ptr + tid)
    x = permute2d_col(x, permute_tid, BLOCK_SIZE)
    tl.store(y_ptr + tid, x)


BLOCK_SIZE = 32
x = torch.arange(BLOCK_SIZE, device='cuda').to(torch.float16)[None, :] * torch.ones((BLOCK_SIZE, BLOCK_SIZE), device="cuda")
y = torch.empty((BLOCK_SIZE, BLOCK_SIZE), device='cuda', dtype=torch.float16)
kernel[(1,)](x, y, BLOCK_SIZE)
print(y)
