# kernels/naive.py
from cutlass import cute

TILE_X = 16
TILE_Y = 16
BLOCKSIZE = 16

@cute.kernel
def smem_sliding_kernel(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    tx, ty, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    bdx, bdy, _ = cute.arch.block_dim()

    r = by * bdy + ty
    c = bx * bdx + tx

    M, N = C.shape
    K = A.shape[1]
    
    
    A_sptr = cute.arch.alloc_smem(cute.Float32,BLOCKSIZE * BLOCKSIZE) # A shared memory
    B_sptr = cute.arch.alloc_smem(cute.Float32,BLOCKSIZE * BLOCKSIZE) # B shared memory
    layout_sq = cute.make_layout((BLOCKSIZE, BLOCKSIZE), stride=(BLOCKSIZE, 1))
    A_s = cute.make_tensor(A_sptr, layout_sq)
    B_s = cute.make_tensor(B_sptr, layout_sq)
    n_blocks = (K + BLOCKSIZE - 1) // BLOCKSIZE
    
    acc = cute.Float32(0.0)
    for block_idx in range(n_blocks):
        k_g = block_idx * BLOCKSIZE + tx                
        A_s[ty,tx] = A[r, k_g]
        B_s[ty,tx] = B[k_g, c]
        cute.arch.sync_threads()
        
        for k in range(BLOCKSIZE):
            acc += A_s[ty, k] * B_s[k, tx]
        
        cute.arch.sync_threads()
            
    if r < M and c < N:
        C[r, c] = acc


@cute.jit
def smem_sliding(A_d: cute.Tensor, B_d: cute.Tensor, C_d: cute.Tensor):
    M, N = C_d.shape
    grid_x = (N + TILE_X - 1) // TILE_X    
    grid_y = (M + TILE_Y - 1) // TILE_Y    
    kernel = smem_sliding_kernel(A_d, B_d, C_d)
    kernel.launch(grid=(grid_x, grid_y, 1), block=(TILE_X, TILE_Y, 1))
    