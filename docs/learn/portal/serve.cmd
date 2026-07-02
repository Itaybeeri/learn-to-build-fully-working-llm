@echo off
REM ============================================================
REM  Learn to Build an LLM Learning Portal - one-click launcher
REM  Double-click this file (or run it) to serve the portal at
REM  http://localhost:8000/ and open it in your browser.
REM  Close this window (or press Ctrl+C) to stop the server.
REM ============================================================
setlocal
cd /d "%~dp0"
set PORT=8000

echo.
echo   Learn to Build an LLM Learning Portal
echo   Serving this folder at http://localhost:%PORT%/
echo   Keep this window open while you learn. Close it to stop.
echo.

REM Open the browser (fires async; the server below comes up fast).
start "" "http://localhost:%PORT%/index.html"

REM Prefer the Windows "py" launcher; fall back to "python".
where py >nul 2>nul
if %errorlevel%==0 (
    py -m http.server %PORT%
) else (
    python -m http.server %PORT%
)
