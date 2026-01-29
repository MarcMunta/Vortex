@echo off
setlocal
echo [stop] Closing terminals and processes...
rem Kill common AI/runtime processes
for %%P in (python.exe pythonw.exe) do taskkill /F /T /IM %%P >nul 2>&1
rem Close terminals
for %%T in (cmd.exe powershell.exe pwsh.exe WindowsTerminal.exe wt.exe) do taskkill /F /T /IM %%T >nul 2>&1
endlocal
