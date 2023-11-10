echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v231\fluent/ntbin/win64/winkill.exe"

"C:\PROGRA~1\ANSYSI~1\v231\fluent\ntbin\win64\tell.exe" Guilherme 55664 CLEANUP_EXITING
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 23188) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 11432) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 3732) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 20804) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 15424) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 22696) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 6360) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 20652) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 2744) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 23492)
del "H:\Meu Drive\TCC\Programming\cnn-hypersonic\pyFluent\cleanup-fluent-Guilherme-2744.bat"
