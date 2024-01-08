echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v231\fluent/ntbin/win64/winkill.exe"

"C:\PROGRA~1\ANSYSI~1\v231\fluent\ntbin\win64\tell.exe" Guilherme 56788 CLEANUP_EXITING
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 18048) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 19668) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 21236) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 13960) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 9664) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 13672) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 22304) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 21980) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 18084) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 5672)
del "H:\Meu Drive\TCC\Programming\cnn-hypersonic\cleanup-fluent-Guilherme-18084.bat"
