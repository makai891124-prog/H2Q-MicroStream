param(
    [string]$Source = "data/open_corpus/open_corpus.bin",
    [int]$SeqLen = 1024,
    [int]$Segments = 3,
    [int]$StepsPerSegment = 30000,
    [int]$TelemetryEvery = 1000,
    [double]$MainLr = 0.0003,
    [double]$CtrlLr = 0.0003
)

$ErrorActionPreference = "Stop"

Write-Host "[pipeline] start main24h"
powershell -ExecutionPolicy Bypass -File .\start_longrun_24h.ps1 `
    -Source $Source `
    -SeqLen $SeqLen `
    -Lr $MainLr `
    -Segments $Segments `
    -StepsPerSegment $StepsPerSegment `
    -TelemetryEvery $TelemetryEvery `
    -RunName "main24h"

Write-Host "[pipeline] start ctrl24h"
powershell -ExecutionPolicy Bypass -File .\start_longrun_24h.ps1 `
    -Source $Source `
    -SeqLen $SeqLen `
    -Lr $CtrlLr `
    -Segments $Segments `
    -StepsPerSegment $StepsPerSegment `
    -TelemetryEvery $TelemetryEvery `
    -RunName "ctrl24h"

Write-Host "[pipeline] summarize acceptance and strict compare"
python .\summarize_acceptance_and_compare.py

Write-Host "[pipeline] append final analysis report"
python .\append_24h_final_report.py

Write-Host "[pipeline] done"
