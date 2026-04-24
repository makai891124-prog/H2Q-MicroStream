param(
    [string]$Source = "data/open_corpus/open_corpus.bin",
    [int]$SeqLen = 1024,
    [double]$Lr = 0.0003,
    [int]$Segments = 3,
    [int]$StepsPerSegment = 30000,
    [int]$TelemetryEvery = 1000,
    [string]$RunName = "longrun_24h"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $Source)) {
    throw "Source not found: $Source"
}

$lastCkpt = "${RunName}_last.pt"
$bestCkpt = "${RunName}_best.pt"
$emerCkpt = "${RunName}_emergency.pt"

Write-Host "[start] run=$RunName segments=$Segments steps_per_segment=$StepsPerSegment"

for ($i = 1; $i -le $Segments; $i++) {
    $telemetry = "evolution_telemetry_${RunName}_seg${i}.csv"
    $acceptJson = "acceptance_${RunName}_seg${i}.json"
    $acceptMd = "acceptance_${RunName}_seg${i}.md"

    $cmd = @(
        "local_evolution_daemon.py",
        "--source", $Source,
        "--seq-len", "$SeqLen",
        "--lr", "$Lr",
        "--telemetry-every", "$TelemetryEvery",
        "--print-every", "$TelemetryEvery",
        "--svd-every", "$TelemetryEvery",
        "--cache-clear-every", "10000",
        "--telemetry-csv", $telemetry,
        "--best-checkpoint", $bestCkpt,
        "--emergency-checkpoint", $emerCkpt,
        "--final-checkpoint", $lastCkpt,
        "--max-steps", "$StepsPerSegment"
    )

    if (Test-Path $lastCkpt) {
        $cmd += @("--resume", $lastCkpt)
    }

    Write-Host "[segment $i] python $($cmd -join ' ')"
    python @cmd

    Write-Host "[segment $i] dynamic acceptance"
    python dynamic_acceptance.py `
        --baseline baseline_snapshot.json `
        --telemetry $telemetry `
        --hypotheses autopilot_hypotheses.jsonl `
        --seq-len $SeqLen `
        --output-json $acceptJson `
        --output-md $acceptMd
}

Write-Host "[done] run=$RunName"
Write-Host "[done] final checkpoint: $lastCkpt"
