import torch
from cutlass import cute
from cutlass.cute.runtime import from_dlpack
import statistics as stats

import kernels

torch.backends.cuda.matmul.allow_tf32 = False  # disables TensorFloat-32 on A100+/Hopper
torch.set_float32_matmul_precision("high")  # prefer high precision

import warnings

warnings.filterwarnings("ignore", message="This loop is no longer unrolled*")


def gflops(h, k, w, ms):
    # 2 * H*K*W floating ops
    flop = 2.0 * h * k * w
    return flop / (ms * 1e-3) / 1e9


if __name__ == "__main__":
    torch.manual_seed(1995)
    print("Matrix Multiplication Benchmark:")
    print("-----------------------------------------------------------")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print("Device:", props.name)
    print("Compute capability:", f"{props.major}.{props.minor}")
    print("Warp size:", props.warp_size)
    print("Multiprocessors:", props.multi_processor_count)
    print("Max threads per SM:", props.max_threads_per_multi_processor)
    print("L2 cache size (bytes):", props.L2_cache_size)
    print("Shared memory per block (bytes):", props.shared_memory_per_block)
    print("Shared memory per multiprocessor (bytes):", props.shared_memory_per_multiprocessor)
    M, N, K = 2048, 2048, 2048

    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)

    A_d = from_dlpack(A, assumed_align=16)
    B_d = from_dlpack(B, assumed_align=16)
    C_d = from_dlpack(C, assumed_align=16)

    def torch_kernel():
        return A @ B

    ground_truth = torch_kernel()  # warmup

    t_times = kernels.time.time_cuda_fn(torch_kernel, warmup=5, iters=15)
    t_mean, t_std = stats.mean(t_times), stats.pstdev(t_times)

    print(f"{'Kernel':<18} {'Time [ms]':>12} {'GFLOP/s':>12} {'% of Torch':>12}")
    print("-" * 56)

    # Torch baseline row
    torch_gflops = gflops(M, K, N, t_mean)
    print(f"{'Torch':<18} {t_mean:12.3f} {torch_gflops:12.2f} {'100.00':>12}")

    for kernel_name in kernels.kernel_map:
        kernel_pack = kernels.kernel_map[kernel_name]
        compiled_kernel = cute.compile(kernel_pack["kernel"], A_d, B_d, C_d, **kernel_pack["args"])

        challenger_kernel = lambda: compiled_kernel(A_d, B_d, C_d, **kernel_pack["args"])
        challenger_kernel()
        torch.cuda.synchronize()
        torch.testing.assert_close(C, ground_truth, rtol=0.0, atol=7.5e-2)

        k_times = kernels.time.time_cuda_fn(challenger_kernel, warmup=5, iters=15)
        k_mean, k_std = stats.mean(k_times), stats.pstdev(k_times)
        k_gflops = gflops(M, K, N, k_mean)
        pct = t_mean / k_mean * 100
        print(f"{kernel_name:<18} {k_mean:12.3f} {k_gflops:12.2f} {pct:12.2f}")
