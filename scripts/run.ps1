# scripts/run.ps1 — wrapper that pulls API keys from User-scope env and
# delegates to a harness CLI module. Avoids per-shell `$env:...=...`
# boilerplate and keeps secrets out of shell history.
#
# Usage:
#   .\scripts\run.ps1 -Module harness.runner --model claude-haiku-4-5 --provider anthropic --mode zero-shot --limit 5
#   .\scripts\run.ps1 -Module harness.scoring results/claude-haiku-4-5-zero-shot-smoke.jsonl

[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [string]$Module,

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Rest
)

$env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY', 'User')
$env:OPENAI_API_KEY    = [Environment]::GetEnvironmentVariable('OPENAI_API_KEY',    'User')

if (-not $env:ANTHROPIC_API_KEY -and -not $env:OPENAI_API_KEY) {
    Write-Error "Neither ANTHROPIC_API_KEY nor OPENAI_API_KEY found at User scope. Set them with [Environment]::SetEnvironmentVariable(..., 'User') and reopen this shell."
    exit 1
}

# Prefer the venv interpreter when present
$python = if (Test-Path .\.venv\Scripts\python.exe) { '.\.venv\Scripts\python.exe' } else { 'python' }

& $python -m $Module @Rest
exit $LASTEXITCODE
