from . import time
from .K01_naive import naive_gemm
from .K02_naive_coalesced import naive_coalesced_gemm
from .K03_sliding_smem import smem_sliding

kernel_map = {
    "naive": {"kernel": naive_gemm, "args": {}},
    "naive_coalesced": {"kernel": naive_coalesced_gemm, "args": {}},
    "sliding": {"kernel": smem_sliding, "args": {}},
}
