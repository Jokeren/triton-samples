import torch
import triton
import triton.language as tl


@triton.jit
def kernel2(in_ptr0, out_ptr5, xnumel: tl.constexpr, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    tmp2 = tl.load(in_ptr0 + xindex, xmask)
    tmp39 = tl.libdevice.log(tmp2)
    tl.store(out_ptr5 + xindex, tmp39, xmask)


def call():
    arg1 = torch.abs(torch.randn((204, 204, 26), device="cuda", dtype=torch.float64))
    buf29 = torch.abs(torch.randn((204, 204, 26), device="cuda", dtype=torch.float64))
    kernel2[(triton.cdiv(1082016, 1024),)](arg1, buf29, 1082016, 1024)
    print(buf29)


call()
