import torch
  
import triton
import triton.language as tl

@triton.jit
def binary_search(
    x,
    codebook,
    BLOCK_SIZE: tl.constexpr,
    CODE_BOOK_SIZE: tl.constexpr,
):
    cmp = (x[:,None] > codebook[None,:]).to(tl.int32)
    idx = tl.sum(cmp, axis=1)
    seq = tl.arange(0, CODE_BOOK_SIZE)[None,:].broadcast_to((BLOCK_SIZE, CODE_BOOK_SIZE))
    codebook = codebook[None,:].broadcast_to((BLOCK_SIZE, CODE_BOOK_SIZE))
    idx = idx[:,None].broadcast_to((BLOCK_SIZE, CODE_BOOK_SIZE))
    res = tl.sum((seq == idx) * codebook, axis=1)
    return res


@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    c_ptr,
    BLOCK_SIZE: tl.constexpr,
    CODE_BOOK_SIZE: tl.constexpr
):
    boffs = tl.arange(0, BLOCK_SIZE)
    coffs = tl.arange(0, CODE_BOOK_SIZE)
    x = tl.load(x_ptr + boffs)
    c = tl.load(c_ptr + coffs)
    x = binary_search(x, c, BLOCK_SIZE, CODE_BOOK_SIZE)
    tl.store(y_ptr + boffs, x)


BLOCK_SIZE = 128
CODE_BOOK_SIZE = 16
codebook = torch.randint(0, 16, size=(CODE_BOOK_SIZE,), device="cuda", dtype=torch.int32)
data = torch.randint(0, 16, size=(BLOCK_SIZE,), device="cuda", dtype=torch.int32)
res = torch.zeros((BLOCK_SIZE,), device="cuda", dtype=torch.int32)
codebook, _ = torch.sort(codebook)
# Suppose the last element of the code book is always greater than the data
codebook[-1] = 16
kernel[(1,)](data, res, codebook, BLOCK_SIZE, CODE_BOOK_SIZE)
ref_idx = torch.searchsorted(codebook, data)
ref = codebook[ref_idx]


print("codebook: ", codebook)
print("data: ", data)
print("res: ", res)
print("ref: ", ref)
