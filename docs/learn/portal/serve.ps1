# Learn to Build an LLM Learning Portal - one-click launcher (PowerShell)
# Run from a PowerShell prompt:  .\serve.ps1
# Serves this folder at http://localhost:8000/ and opens the browser.
# Press Ctrl+C to stop the server.

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot
$port = 8000

Write-Host ""
Write-Host "  Learn to Build an LLM Learning Portal" -ForegroundColor Cyan
Write-Host "  Serving this folder at http://localhost:$port/"
Write-Host "  Keep this window open while you learn. Ctrl+C to stop."
Write-Host ""

# Open the browser (server below comes up fast).
Start-Process "http://localhost:$port/index.html"

# Prefer the "py" launcher; fall back to "python".
if (Get-Command py -ErrorAction SilentlyContinue) {
    py -m http.server $port
} else {
    python -m http.server $port
}
