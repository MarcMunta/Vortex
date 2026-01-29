@echo off
setlocal
cd /d "%~dp0"
set "TASK=C3RNT2_Restart"
for /f "usebackq delims=" %%t in (`powershell -NoProfile -Command "(Get-Date).AddSeconds(5).ToString('HH:mm')"`) do set "RUNAT=%%t"
for /f "usebackq delims=" %%d in (`powershell -NoProfile -Command "(Get-Date).ToString('MM/dd/yyyy')"`) do set "RUNDATE=%%d"
rem Schedule start after stop closes terminals
schtasks /Create /TN "%TASK%" /TR "\"%~dp0start.bat\"" /SC ONCE /ST %RUNAT% /SD %RUNDATE% /F /Z >nul
call "%~dp0stop.bat"
endlocal
