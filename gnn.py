import torch
import triton
import triton.language as tl

@triton.jit
def get_kernel(
    x_ptr,
    index_ptr,
    y_ptr,
    z_ptr,
    N: tl.constexpr,
    C: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    index_offsets = index_ptr + block_start + offsets
    index = tl.load(index_offsets, mask=mask)
    mask = index < N

    offsets = index[:, None] * C + tl.arange(0, C)[None, :]
    x_offsets = x_ptr + offsets
    y_offsets = y_ptr + offsets
    x = tl.load(x_offsets)
    y = tl.load(y_offsets)

    norm = tl.sum(x, axis=1)
    z_offsets = z_ptr + index
    tl.store(z_offsets, norm, mask=mask)

BLOCK_SIZE = 32

N = 32
# correct
#C = 4
# wrong, "Unknown modifier '.v3'", "Illegal vector size: 3"
#C = 12
# wrong, result = 36, the last elements in each row are not taken into account
C = 9

index = torch.arange(N, device='cuda', dtype=torch.int)
x = torch.stack([torch.arange(C, dtype=torch.int)+1 for _ in range(N)], dim=0).contiguous().cuda()
y = torch.zeros((N, C), device='cuda', dtype=torch.int)
z = torch.zeros((N, ), device='cuda', dtype=torch.int)

grid = lambda meta: (triton.cdiv(N, BLOCK_SIZE),)
get_kernel[grid](x, index, y, z, N, C, BLOCK_SIZE)

print("x:\n", x)
print("z:\n", z)
