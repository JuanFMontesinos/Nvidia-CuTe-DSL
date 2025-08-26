import statistics as stats

import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from ppp_utils import time_cuda_fn


@cute.kernel
def matrix_multiplication_kernel(
    M: cute.Tensor,  # (H, K)
    N: cute.Tensor,  # (K, W)
    P: cute.Tensor,  # (H, W)
):
    # Thread/block indices
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    bdim_x, bdim_y, _ = cute.arch.block_dim()

    row = by * bdim_y + ty
    col = bx * bdim_x + tx

    if row < P.shape[0] and col < P.shape[1]:
        acc = cute.Float32(0.0)  # accumulate in fp32 for better precision
        for k in range(M.shape[1]):
            acc += (M[row, k]) * (N[k, col]).to(cute.Float32)
        # store back in fp16
        P[row, col] = acc.to(cute.Float16)


@cute.jit
def matrix_multiplication(M_d: cute.Tensor, N_d: cute.Tensor, P_d: cute.Tensor):
    block_x = 16
    block_y = 16

    # Derive launch dims from the output P
    grid_x = (P_d.shape[1] + block_x - 1) // block_x
    grid_y = (P_d.shape[0] + block_y - 1) // block_y

    print("[DSL INFO] Input tensors:")
    print(f"[DSL INFO]   M_d = {M_d}")
    print(f"[DSL INFO]   N_d = {N_d}")
    print(f"[DSL INFO]   P_d = {P_d}")

    kernel = matrix_multiplication_kernel(M_d, N_d, P_d)
    kernel.launch(
        grid=(grid_x, grid_y, 1),
        block=(block_x, block_y, 1),
    )


def gflops(h, k, w, ms):
    # 2 * H*K*W floating ops
    flop = 2.0 * h * k * w
    return flop / (ms * 1e-3) / 1e9


if __name__ == "__main__":
    torch.manual_seed(0)

    # HxK @ KxW -> HxW
    H, K, W = 2048, 2048, 2048

    m = torch.randn(H, K, device="cuda", dtype=torch.float16)
    n = torch.randn(K, W, device="cuda", dtype=torch.float16)
    p = torch.empty(H, W, device="cuda", dtype=torch.float16)

    m_d = from_dlpack(m, assumed_align=16)
    n_d = from_dlpack(n, assumed_align=16)
    p_d = from_dlpack(p, assumed_align=16)

    # Compile
    matmul_ = cute.compile(matrix_multiplication, m_d, n_d, p_d)

    # Define callables
    def run_kernel():
        matmul_(m_d, n_d, p_d)

    def run_torch():
        # compute in fp32, store fp16 (matches your ref)
        p_ref = (m.float() @ n.float()).half()
        torch.cuda.current_stream().synchronize()
        return p_ref

    # Verify once (outside timing)
    run_kernel()
    ref = (m.float() @ n.float()).half()
    torch.testing.assert_close(p, ref, rtol=1e-2, atol=1e-2)

    # Time kernel
    k_times = time_cuda_fn(run_kernel, warmup=5, iters=50)
    k_mean, k_std = stats.mean(k_times), stats.pstdev(k_times)
    print(f"[Kernel] mean {k_mean:.3f} ms  std {k_std:.3f} ms  {gflops(H, K, W, k_mean):.2f} GFLOP/s")

    out_holder = torch.empty_like(p)

    def run_torch_noalloc():
        out_holder.copy_((m.float() @ n.float()).half(), non_blocking=True)

    t_times = time_cuda_fn(run_torch_noalloc, warmup=5, iters=50)
    t_mean, t_std = stats.mean(t_times), stats.pstdev(t_times)
    print(f"[Torch @] mean {t_mean:.3f} ms  std {t_std:.3f} ms  {gflops(H, K, W, t_mean):.2f} GFLOP/s")
