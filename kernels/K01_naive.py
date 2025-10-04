# kernels/naive.py
from cutlass import cute

TILE_X = 16
TILE_Y = 16


@cute.kernel
def naive_gemm_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    bdx, bdy, _ = cute.arch.block_dim()

    c = by * bdy + ty
    r = bx * bdx + tx  

    M, N = C.shape
    K = A.shape[1]

    if r < M and c < N:
        acc = cute.Float32(0.0) 
        for k in range(K):
            acc += A[r, k] * B[k, c]
        C[r, c] = acc


@cute.jit
def naive_gemm(A_d: cute.Tensor, B_d: cute.Tensor, C_d: cute.Tensor):
    M, N = C_d.shape
    grid_x = (M + TILE_X - 1) // TILE_X    # rows via x
    grid_y = (N + TILE_Y - 1) // TILE_Y    # cols via y
    kernel = naive_gemm_kernel(A_d, B_d, C_d)
    kernel.launch(grid=(grid_x, grid_y, 1), block=(TILE_X, TILE_Y, 1))