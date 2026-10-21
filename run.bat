@echo off
REM ---------------------------------------------------------------
REM  Launcher for: pick_risk.py / verify.py
REM  Kept pure ASCII on purpose -- a .bat with Chinese text gets
REM  mangled by the cmd.exe OEM codepage (GBK) and the lines break.
REM  All Chinese output comes from the Python scripts instead.
REM ---------------------------------------------------------------
setlocal
cd /d "%~dp0"

echo ============================================
echo  Step 1/3  Self-check (independent recompute)
echo ============================================
python verify.py
if errorlevel 1 (
    echo.
    echo [FAIL] Self-check did not pass. Fix the FAIL lines above first.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Step 2/3  Root-cause report + flags
echo ============================================
python pick_risk.py
if errorlevel 1 (
    echo.
    echo [FAIL] pick_risk.py crashed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo  Step 3/3  Regenerate dispatch list (stale only)
echo ============================================
python pick_risk.py flag --only-stale

echo.
echo Done.
echo   report : %~dp0out\summary.txt
echo   flags  : %~dp0out\risk_flags.csv
pause
endlocal
