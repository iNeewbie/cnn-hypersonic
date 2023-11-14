echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v231\fluent/ntbin/win64/winkill.exe"

"C:\PROGRA~1\ANSYSI~1\v231\fluent\ntbin\win64\tell.exe" Guilherme 59865 CLEANUP_EXITING
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 2536) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 15440) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 17452) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 12000) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 728) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 16276) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 17504) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 15052) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 17436) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 1844)
del "H:\Meu Drive\TCC\Programming\cnn-hypersonic\pyFluent\cleanup-fluent-Guilherme-17436.bat"
