import torch
import triton
import triton.language as tl


@triton.jit
def kernel0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK, 1])
    xmask = xindex < xnumel
    rbase = tl.reshape(tl.arange(0, RBLOCK), [1, RBLOCK])
    _tmp1 = tl.zeros([XBLOCK, RBLOCK], tl.float32) + float("-inf")
    _tmp1_index = tl.zeros([XBLOCK, RBLOCK], tl.int32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + r0, rmask, eviction_policy='evict_last').to(tl.float32)
        _tmp1_index = tl.where(xmask & rmask & (_tmp1 < tmp0),  rindex, _tmp1_index)
        _tmp1 = tl.where(xmask & rmask & (_tmp1 < tmp0), tmp0, _tmp1)
    _tmp1_index_reduce = tl.reshape(tl.argmax(_tmp1, 1), [XBLOCK, 1]).to(tl.int32)
    _tmp1_index_mask = (tl.arange(0, RBLOCK)[None, :] == _tmp1_index_reduce)
    tl.store(out_ptr0 + tl.where(_tmp1_index_mask, 0, 0), value=_tmp1_index, mask=_tmp1_index_mask&xmask)
    #tmp1 = tl.reshape(tl.sum(tl.where(_tmp1_index_mask, _tmp1_index, 0), 1), [XBLOCK, 1])
    #tl.store(out_ptr0 + 0 + tl.zeros(tmp1.shape, tl.int32), tmp1, xmask)


def call():
    x = torch.randn((1, 8192), device="cuda")
    y = torch.zeros((1), device="cuda")
    kernel0[(1,)](x, y, 1, 8192, XBLOCK=1, RBLOCK=128)
    print(torch.argmax(x))
    print(y)


call()
