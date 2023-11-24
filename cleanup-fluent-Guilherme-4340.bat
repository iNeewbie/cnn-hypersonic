echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v231\fluent/ntbin/win64/winkill.exe"

"C:\PROGRA~1\ANSYSI~1\v231\fluent\ntbin\win64\tell.exe" Guilherme 64664 CLEANUP_EXITING
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 17944) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 9412) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 3044) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 18296) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 14624) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 3900) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 20820) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 3612) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 4340) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 5508)
del "H:\Meu Drive\TCC\Programming\cnn-hypersonic\cleanup-fluent-Guilherme-4340.bat"
