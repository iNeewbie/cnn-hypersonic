echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="C:\PROGRA~1\ANSYSI~1\v231\fluent/ntbin/win64/winkill.exe"

"C:\PROGRA~1\ANSYSI~1\v231\fluent\ntbin\win64\tell.exe" Guilherme 58759 CLEANUP_EXITING
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 10800) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 11632) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 21052) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 17140) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 12844) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 908) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 10256) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 20020) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 17692) 
if /i "%LOCALHOST%"=="Guilherme" (%KILL_CMD% 1800)
del "H:\Meu Drive\TCC\Programming\cnn-hypersonic\pyFluent\cleanup-fluent-Guilherme-17692.bat"
