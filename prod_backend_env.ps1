Param(
    [switch]$Rollback,
    [int]$MinSeq = 256,
    [string]$Python = "python",
    [string]$Script = "",
    [string[]]$ScriptArgs = @()
)

$ErrorActionPreference = "Stop"

function Show-EnvSummary {
    Write-Host "Backend env in current shell:" -ForegroundColor Cyan
    $keys = @(
        "BINARY_STA_FORCE_TRAIN_PACKBITS",
        "BINARY_STA_PACKBITS_INFER_CUDA_EXT",
        "BINARY_STA_CUDA_EXT_MODE",
        "BINARY_STA_CUDA_EXT_MIN_SEQ",
        "BINARY_STA_CUDA_EXT_PROFILE",
        "BINARY_STA_DISABLE_CUDA_EXT"
    )
    foreach ($k in $keys) {
        $v = (Get-Item -Path ("Env:{0}" -f $k) -ErrorAction SilentlyContinue).Value
        Write-Host ("  {0}={1}" -f $k, $v)
    }
}

if ($Rollback) {
    # One-click rollback: force pure packbits behavior.
    $env:BINARY_STA_FORCE_TRAIN_PACKBITS = "1"
    $env:BINARY_STA_PACKBITS_INFER_CUDA_EXT = "0"
    $env:BINARY_STA_CUDA_EXT_MODE = "always"
    $env:BINARY_STA_CUDA_EXT_PROFILE = "fast"
    $env:BINARY_STA_DISABLE_CUDA_EXT = "1"
    Remove-Item Env:BINARY_STA_CUDA_EXT_MIN_SEQ -ErrorAction SilentlyContinue

    Write-Host "[ROLLBACK] Switched to pure packbits mode." -ForegroundColor Yellow
} else {
    # Production default:
    # - Training fixed to packbits.
    # - Inference can use cuda_ext only for long sequences.
    $env:BINARY_STA_FORCE_TRAIN_PACKBITS = "1"
    $env:BINARY_STA_PACKBITS_INFER_CUDA_EXT = "1"
    $env:BINARY_STA_CUDA_EXT_MODE = "infer_long"
    $env:BINARY_STA_CUDA_EXT_MIN_SEQ = [string][Math]::Max($MinSeq, 1)
    $env:BINARY_STA_CUDA_EXT_PROFILE = "fast"
    $env:BINARY_STA_DISABLE_CUDA_EXT = "0"

    Write-Host "[PROD] Training fixed to packbits; inference long-seq cuda_ext enabled." -ForegroundColor Green
}

Show-EnvSummary

if ($Script -ne "") {
    Write-Host "\nRunning: $Python $Script $($ScriptArgs -join ' ')" -ForegroundColor Cyan
    & $Python $Script @ScriptArgs
    exit $LASTEXITCODE
}
