echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v231\fluent/ntbin/win64/winkill.exe"

"C:\PROGRA~1\ANSYSI~1\v231\fluent\ntbin\win64\tell.exe" Guilherme 63106 CLEANUP_EXITING
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 21384) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 13284) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 20728) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 18728) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 19592) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 5020) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 13944) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 6600) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 3880) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 18392)
del "H:\Meu Drive\TCC\Programming\cnn-hypersonic\pyFluent\cleanup-fluent-Guilherme-3880.bat"
