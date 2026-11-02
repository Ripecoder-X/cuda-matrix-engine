# Phase 7 Profiling Commands

These are the exact commands that should be run to complete nsys profiling for the CUDA Matrix Engine:

## Kernel Profiling Commands

### 1. Profile Naive Kernel
```
nsys profile --output=results/profiles/naive_kernel.nsys --target-processes=benchmark_main.exe ./build/windows-release/benchmarks/benchmark_main.exe --kernel-only --size 512 --algorithm=CUDA_NAIVE
```

### 2. Profile Optimized Kernel (Best Performing)
```
nsys profile --output=results/profiles/optimized_kernel.nsys --target-processes=benchmark_main.exe ./build/windows-release/benchmarks/benchmark_main.exe --kernel-only --size 512 --algorithm=CUDA_OPTIMIZED --tile-size=32
```

### 3. Extract Detailed Analysis
```
nsys stats --report=kernel-details --output=results/profiles/kernel_analysis.txt results/profiles/optimized_kernel.nsys
```

## Target Metrics to Analyze

### Memory Performance:
- Shared memory usage and bank conflicts  
- Global memory access patterns
- Memory throughput and bandwidth utilization

### Compute Performance:
- Occupancy (thread blocks per SM)
- Register usage per thread
- Instruction mixes and pipeline efficiency  

### Optimization Opportunities:
- Cache hit ratios
- Load balancing between threads
- Memory coalescing efficiency

## Output Files
All profiling data will be stored in: `results/profiles/` directory with extensions:
- `.nsys` - Raw profiler capture files  
- `.txt` - Extracted statistics and analysis