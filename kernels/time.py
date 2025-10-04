import torch

def time_cuda_fn(fn, *args, warmup=5, iters=10):
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    for _ in range(iters):
        start.record()
        fn(*args)
        end.record()
        end.synchronize()            # wait for this iteration
        times.append(start.elapsed_time(end))  # milliseconds
    return times