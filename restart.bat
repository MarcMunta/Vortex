@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"

rem Load env defaults from .env (do not override already-defined vars)
if exist "%ROOT%.env" (
  for /f "usebackq tokens=1* delims== eol=#" %%A in ("%ROOT%.env") do (
    if not "%%A"=="" (
      if not defined %%A set "%%A=%%B"
    )
  )
)

rem Prefer PowerShell 7 (pwsh) when available
set "PS_EXE="
where /q pwsh.exe && set "PS_EXE=pwsh"
if not defined PS_EXE where /q powershell.exe && set "PS_EXE=powershell"
if not defined PS_EXE (
  echo [restart] ERROR: PowerShell not found in PATH.
  echo [restart] Install PowerShell 7: pwsh.exe, or Windows PowerShell: powershell.exe.
  exit /b 1
)

%PS_EXE% -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%ROOT%stop.ps1"
set "STOP_RC=%errorlevel%"
if not "%STOP_RC%"=="0" (
  echo [restart] WARN: stop returned errorlevel %STOP_RC%.
)

%PS_EXE% -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%ROOT%run.ps1" %*
exit /b %errorlevel%
