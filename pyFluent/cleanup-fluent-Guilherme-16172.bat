echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v231\fluent/ntbin/win64/winkill.exe"

"C:\PROGRA~1\ANSYSI~1\v231\fluent\ntbin\win64\tell.exe" Guilherme 62934 CLEANUP_EXITING
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 12692) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 13280) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 6308) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 13340) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 15160) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 3832) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 16516) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 11864) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 16172) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 16800)
del "H:\Meu Drive\TCC\Programming\cnn-hypersonic\pyFluent\cleanup-fluent-Guilherme-16172.bat"
