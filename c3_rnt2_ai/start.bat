@echo off
setlocal
cd /d "%~dp0"
if not defined C3RNT2_PROFILE set "C3RNT2_PROFILE=rtx4080_16gb_vortexx_next"
set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"
set "CMD=%*"
if "%CMD%"=="" set "CMD=chat"
echo [start] Vortex (C3RNT2_PROFILE=%C3RNT2_PROFILE%)
%PYTHON% -m vortex %CMD%
endlocal
