# Batch 7.1 Windows driver for the same captures as scripts/profile.sh.
param()

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

$Preset = if ($env:CME_PRESET) { $env:CME_PRESET } else { "windows-release" }

function Find-Tool {
    param([string]$Name, [string[]]$Candidates)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($c in $Candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

$Nsys = if ($env:CME_NSYS) { $env:CME_NSYS } else {
    Find-Tool "nsys" @(
        "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.3.1\target-windows-x64\nsys.exe",
        "C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe"
    )
}
$Ncu = if ($env:CME_NCU) { $env:CME_NCU } else {
    Find-Tool "ncu" @(
        "C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.1\ncu.bat"
    )
}

if (-not $Nsys) { Write-Error "nsys not found. Set CME_NSYS."; exit 1 }
if (-not $Ncu) { Write-Error "ncu not found. Set CME_NCU."; exit 1 }

Write-Host "Configuring and building preset: $Preset"
& cmake --preset $Preset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& cmake --build --preset $Preset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$BenchBinary = Join-Path $RootDir "build\$Preset\benchmarks\benchmark_main.exe"
if (-not (Test-Path $BenchBinary)) {
    Write-Error "Benchmark binary not found at $BenchBinary"
    exit 1
}

$ProfileDir = Join-Path $RootDir "results\profiles"
New-Item -ItemType Directory -Force -Path $ProfileDir | Out-Null

Write-Host ""
Write-Host "=== Nsight Systems: end-to-end size 2048 ==="
Write-Host "nsys: $Nsys"
& $Nsys --version
$NsysOut = Join-Path $ProfileDir "nsys_end_to_end_2048"
& $Nsys profile --force-overwrite=true --stats=true --trace=cuda,nvtx,osrt -o $NsysOut `
    $BenchBinary --size 2048 --end-to-end --warmup 1 --iterations 3 `
    --output (Join-Path $ProfileDir "temp_e2e.csv")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$NsysRep = "$NsysOut.nsys-rep"
if (Test-Path $NsysRep) {
    & $Nsys stats --report cuda_gpu_kern_sum --format csv -o (Join-Path $ProfileDir "nsys_end_to_end_2048_kern") $NsysRep
    & $Nsys stats --report cuda_gpu_mem_time_sum --format csv -o (Join-Path $ProfileDir "nsys_end_to_end_2048_mem") $NsysRep
}

$NcuMetrics = @(
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "smsp__average_warps_issue_stalled_long_scoreboard.pct",
    "smsp__average_warps_issue_stalled_short_scoreboard.pct",
    "smsp__average_warps_issue_stalled_barrier.pct",
    "smsp__average_warps_issue_stalled_membar.pct",
    "smsp__average_warps_issue_stalled_mio_throttle.pct",
    "smsp__average_warps_issue_stalled_lg_throttle.pct",
    "smsp__average_warps_issue_stalled_misc.pct",
    "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
    "launch__registers_per_thread",
    "launch__occupancy_limit_registers",
    "launch__occupancy_limit_shared_mem",
    "launch__occupancy_limit_warps",
    "launch__grid_size",
    "launch__block_size",
    "launch__shared_mem_per_block_allocated"
) -join ","

Write-Host ""
Write-Host "ncu: $Ncu"
& $Ncu --version

function Invoke-NcuKernel {
    param([string]$Label, [string]$Regex)
    $out = Join-Path $ProfileDir "ncu_${Label}_2048"
    Write-Host ""
    Write-Host "=== Nsight Compute: $Label (size 2048) ==="
    & $Ncu --force-overwrite --target-processes all `
        --kernel-name-base demangled `
        --kernel-name "regex:$Regex" `
        --launch-count 1 `
        --metrics $NcuMetrics `
        --csv `
        --log-file "$out.log" `
        -o $out `
        $BenchBinary --size 2048 --kernel-only --warmup 0 --iterations 1 `
        --output (Join-Path $ProfileDir "temp_kernel_$Label.csv")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $rep = "$out.ncu-rep"
    if (Test-Path $rep) {
        & $Ncu --import $rep --csv --page raw | Out-File -Encoding utf8 "$out.csv"
    }
}

Invoke-NcuKernel "naive" "gemm_naive_kernel"
Invoke-NcuKernel "tiled16" "gemm_tiled_kernel<16>"
Invoke-NcuKernel "tiled32" "gemm_tiled_kernel<32>"
Invoke-NcuKernel "optimized16" "gemm_optimized_kernel<16>"
Invoke-NcuKernel "optimized32" "gemm_optimized_kernel<32>"

Write-Host ""
Write-Host "Profiling complete. Reports in $ProfileDir"
