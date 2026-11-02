param (
    [switch]$Quick
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

$Preset = "windows-release"

Write-Host "Configuring and building preset: $Preset"
& cmake --preset $Preset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& cmake --build --preset $Preset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$BenchBinary = Join-Path $RootDir "build/$Preset/benchmarks/benchmark_main.exe"
if (-not (Test-Path $BenchBinary)) {
    Write-Error "Benchmark binary not found at $BenchBinary"
    exit 1
}

$Timestamp = (Get-Date).ToUniversalTime().ToString("yyyyMMdd_HHmmss")
$ResultsDir = Join-Path $RootDir "results"
if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir | Out-Null
}

$CsvFile = "gemm_$Timestamp.csv"
$CsvOut = Join-Path $ResultsDir $CsvFile
$EnvOut = Join-Path $ResultsDir "$CsvFile.env.txt"

Write-Host "Capturing environment to $EnvOut"
$UtcNow = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$EnvReport = @"
# Environment
Generated on $UtcNow

## Operating system
$([System.Environment]::OSVersion)

## GPU
$(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv)

## Driver
$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)

## NVCC
$(nvcc --version)

## CMake
$(cmake --version)
"@
$EnvReport | Out-File -FilePath $EnvOut -Encoding utf8

$ArgsList = @("--output", $CsvOut)
if ($Quick) {
    $ArgsList += "--quick"
}

Write-Host "Running benchmark: $BenchBinary $ArgsList"
& $BenchBinary @ArgsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ""
Write-Host "Benchmark run complete."
Write-Host "CSV Output: $CsvOut"
Write-Host "Environment: $EnvOut"
