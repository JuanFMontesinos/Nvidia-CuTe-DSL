import statistics as stats

import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
from cutlass import utils  # <- add this import

from ppp_utils import time_cuda_fn

TILE_K = 8

@cute.kernel
def tiled_matrix_multiplication_kernel(
    A: cute.Tensor,  # (M, K)
    B: cute.Tensor,  # (K, N)
    C: cute.Tensor,  # (M, N)
):
    # Constants
    M, K = A.shape
    N = B.shape[1]

    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    bdim_x, bdim_y, _ = cute.arch.block_dim()  # Tile_N, Tile_M

    # your globals
    tx_g = bx * bdim_x + tx  # global x (col)
    ty_g = by * bdim_y + ty  # global y (row)

    smem = utils.SmemAllocator()
    layout_a = cute.make_layout((bdim_y, TILE_K), stride=(TILE_K, 1))   # (tile_M x TILE_K)
    layout_b = cute.make_layout((TILE_K, bdim_x), stride=(bdim_x, 1))   # (TILE_K x tile_N)
    a_s = smem.allocate_tensor(cute.Float16, layout_a, byte_alignment=128)
    b_s = smem.allocate_tensor(cute.Float16, layout_b, byte_alignment=128)

    n_tiles = (K + TILE_K - 1) // TILE_K

    acc = cute.Float32(0.0)

    for k in range(n_tiles):
        kk = tx
        while kk < TILE_K:
            col_g = k * TILE_K + kk
            if ty_g < M and col_g < K:
                a_s[ty, kk] = A[ty_g, k * TILE_K + kk]
            else:
                a_s[ty, kk] = cute.Float16(0)
            kk += bdim_x

        kk = ty
        while kk < TILE_K:
            row_g = k * TILE_K + kk
            if row_g < K and tx_g < N:
                b_s[kk, tx] = B[row_g, tx_g]
            else:
                b_s[kk, tx] = cute.Float16(0)
                
            kk += bdim_y
        cute.arch.sync_threads()

        for kk in range(TILE_K):
            acc += (a_s[ty, kk] * b_s[kk, tx]).to(cute.Float32)
        cute.arch.sync_threads()

    if ty_g < M and tx_g < N:
        C[ty_g, tx_g] = acc.to(cute.Float16)


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


@cute.jit
def tiled_matrix_multiplication(M_d, N_d, P_d):
    block_x = 16  # == tile_N
    block_y = 16  # == tile_M
    grid_x = (P_d.shape[1] + block_x - 1) // block_x
    grid_y = (P_d.shape[0] + block_y - 1) // block_y


    kernel = tiled_matrix_multiplication_kernel(M_d, N_d, P_d)
    kernel.launch(
        grid=(grid_x, grid_y, 1),
        block=(block_x, block_y, 1),
    )


# ---------------------------------------------------------------------------------------------------


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

    # Compile reference (naive) kernel
    matmul_ref = cute.compile(matrix_multiplication, m_d, n_d, p_d)

    # Try compiling the tiled kernel. If the body isn't implemented yet, we’ll skip it gracefully.
    tiled_available = True
    matmul_tiled = cute.compile(tiled_matrix_multiplication, m_d, n_d, p_d)


    # Define callables
    def run_kernel_ref():
        matmul_ref(m_d, n_d, p_d)

    def run_kernel_tiled():
        assert matmul_tiled is not None
        matmul_tiled(m_d, n_d, p_d)

    def run_torch():
        # compute in fp32, store fp16 (matches your ref)
        p_ref = (m.float() @ n.float()).half()
        torch.cuda.current_stream().synchronize()
        return p_ref

    # Verify once (outside timing) for the reference kernel
    run_kernel_ref()
    ref = (m.float() @ n.float()).half()
    torch.testing.assert_close(p, ref, rtol=1e-2, atol=1e-2)

    # Verify tiled vs Torch (only if available)
    if tiled_available:
        p.zero_()
        run_kernel_tiled()
        torch.testing.assert_close(p, ref, rtol=1e-2, atol=1e-2)
        print("[Verify] Tiled kernel matches Torch within rtol=1e-2, atol=1e-2")

    # Time reference kernel
    k_times = time_cuda_fn(run_kernel_ref, warmup=5, iters=50)
    k_mean, k_std = stats.mean(k_times), stats.pstdev(k_times)
    print(f"[Kernel naive] mean {k_mean:.3f} ms  std {k_std:.3f} ms  {gflops(H, K, W, k_mean):.2f} GFLOP/s")

    # Time tiled kernel (if available)
    if tiled_available:
        # ensure output is “owned” similarly
        p.zero_()
        tld_times = time_cuda_fn(run_kernel_tiled, warmup=5, iters=50)
        tld_mean, tld_std = stats.mean(tld_times), stats.pstdev(tld_times)
        print(f"[Kernel tiled] mean {tld_mean:.3f} ms  std {tld_std:.3f} ms  {gflops(H, K, W, tld_mean):.2f} GFLOP/s")
        print(f"[Speedup tiled/naive] {k_mean / tld_mean:.2f}×")

    # Torch baseline without extra allocs
    out_holder = torch.empty_like(p)

    def run_torch_noalloc():
        out_holder.copy_((m.float() @ n.float()).half(), non_blocking=True)

    t_times = time_cuda_fn(run_torch_noalloc, warmup=5, iters=50)
    t_mean, t_std = stats.mean(t_times), stats.pstdev(t_times)
    print(f"[Torch @] mean {t_mean:.3f} ms  std {t_std:.3f} ms  {gflops(H, K, W, t_mean):.2f} GFLOP/s")

    if tiled_available:
        print(f"[Gap vs Torch] tiled/torch: {tld_mean / t_mean:.2f}×  ( <1.0 is faster than Torch )")
    print(f"[Gap vs Torch] naive/torch: {k_mean / t_mean:.2f}×  ( <1.0 is faster than Torch )")
