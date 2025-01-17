import torch
import triton
import triton.language as tl


@triton.jit
def gnn_kernel(
    x_ptr,
    index_ptr,
    y_ptr,
    z_ptr,
    N: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """GNN kernel implementation for graph neural network operations."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    index_offsets = index_ptr + block_start + offsets
    index = tl.load(index_offsets, mask=mask)
    mask = index < N

    offsets = index[:, None] * C + tl.arange(0, C)[None, :]
    x_offsets = x_ptr + offsets
    x = tl.load(x_offsets)

    norm = tl.sum(x, axis=1)
    z_offsets = z_ptr + index
    tl.store(z_offsets, norm, mask=mask)


# Constants
BLOCK_SIZE = 32
N = 32
C = 9  # Note: C=4 works correctly, C=12 causes errors, C=9 has partial results

index = torch.arange(N, device="cuda", dtype=torch.int)
x = (
    torch.stack([torch.arange(C, dtype=torch.int) + 1 for _ in range(N)], dim=0)
    .contiguous()
    .cuda()
)
y = torch.zeros((N, C), device="cuda", dtype=torch.int)
z = torch.zeros((N,), device="cuda", dtype=torch.int)


def grid(meta):
    return (triton.cdiv(N, BLOCK_SIZE),)


gnn_kernel[grid](x, index, y, z, N, C, BLOCK_SIZE)

print("x:\n", x)
print("z:\n", z)
