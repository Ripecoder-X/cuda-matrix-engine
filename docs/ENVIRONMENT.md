# Environment

Recorded by Batch 0.1 on 2026-11-02 from real command output on the owner's machine.

## Operating system

```
$ uname -a
Linux warp.pop-os.org 7.0.11-76070011-generic #202606011647~1784215097~24.04~4974880 SMP PREEMPT_DYNAMIC Thu J x86_64 x86_64 x86_64 GNU/Linux
```

```
$ cat /etc/os-release
NAME="Pop!_OS"
VERSION="24.04 LTS"
ID=pop
```

## CPU and memory

```
$ lscpu
CPU(s):                                  32
Model name:                              Intel(R) Core(TM) i9-14900K
Thread(s) per core:                      2
Core(s) per socket:                      24
Socket(s):                               1
```

```
$ free -g
               total        used        free      shared  buff/cache   available
Mem:              62           9          52           0           4          53
```

Total system RAM: 62 GiB reported by `free -g` (64 GB class).

## GPU

```
$ nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
name, driver_version, memory.total [MiB], compute_cap
NVIDIA GeForce RTX 4070 Ti SUPER, 595.84, 16376 MiB, 8.9
```

- GPU name: NVIDIA GeForce RTX 4070 Ti SUPER
- VRAM: 16376 MiB
- Compute capability: 8.9 (sm_89), read from `nvidia-smi`, not assumed.
- Architecture for `CME_CUDA_ARCHITECTURES`: 89.

## Driver vs toolkit (separate values)

- **Driver version:** 595.84
- **CUDA version reported by the driver:** 13.2 (from the `nvidia-smi` header line)
- **Installed `nvcc` toolkit version:** 12.0 (release 12.0, V12.0.140)

These are different things. The driver supports CUDA 13.2; the installed toolkit is CUDA 12.0.

```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Fri_Jan__6_16:45:21_PST_2023
Cuda compilation tools, release 12.0, V12.0.140
Build cuda_12.0.r12.0/compiler.32267302_0
```

## Other tools

```
$ cmake --version
cmake version 3.28.3
```

```
$ g++ --version
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

```
$ ncu --version
NVIDIA (R) Nsight Compute Command Line Profiler
Copyright (c) 2018-2023 NVIDIA Corporation
Version 2022.4.1.0 (build 32308335) (public-release)
```

```
$ nsys --version
NVIDIA Nsight Systems version 2022.4.2.50-32196742v0
```

```
$ python3 --version
Python 3.12.3
```

## Nominal GPU state at recording time

```
$ nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader,nounits
210, 43, 9.33
```

- SM clock: 210 MHz (idle)
- Temperature: 43 C
- Power draw: 9.33 W

## Windows 11 secondary environment

NOT AVAILABLE. Not tested in this session.
