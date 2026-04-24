# install_build_tools.ps1
# Run this in Admin PowerShell to install VS 2022 Build Tools with C++ workload
# Usage: powershell -ExecutionPolicy Bypass -File .\install_build_tools.ps1

Write-Host "Installing VS 2022 Build Tools with C++ workload..." -ForegroundColor Cyan
Write-Host "(This requires admin privileges and downloads ~2 GB)" -ForegroundColor Yellow

winget install Microsoft.VisualStudio.2022.BuildTools `
    --accept-package-agreements --accept-source-agreements `
    --silent `
    --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --wait"

if ($LASTEXITCODE -eq 0) {
    Write-Host "VS 2022 Build Tools installed successfully." -ForegroundColor Green
} else {
    Write-Host "Install may have failed (exit $LASTEXITCODE). Check Add/Remove Programs." -ForegroundColor Red
    Write-Host "Manual install: https://aka.ms/vs/17/release/vs_buildtools.exe" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "After install, open a NEW terminal and run:" -ForegroundColor Cyan
Write-Host "  cd D:\H2Q-MicroStream"
Write-Host "  python fix_cuda_env.py"
Write-Host "  python cuda_ext_protocol.py"
