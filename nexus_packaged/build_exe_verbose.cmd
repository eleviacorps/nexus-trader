@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0.."
set "LOG_DIR=%~dp0logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%i"
set "LOG_FILE=%LOG_DIR%\build_exe_%TS%.log"

echo [Nexus Build] Streaming verbose output...
echo [Nexus Build] Log file: %LOG_FILE%

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "& { Set-Location '%ROOT%'; & '.\nexus_packaged\build_exe.ps1' -FastBuild -VerboseBuild *>&1 | Tee-Object -FilePath '%LOG_FILE%' }"

echo [Nexus Build] Finished. Exit code: %ERRORLEVEL%
exit /b %ERRORLEVEL%

