@echo off
title Simulation Arena
set "PY=C:\Users\Abhishek Mishra\.workbuddy-ai\binaries\python\envs\simarena\Scripts\python.exe"
cd /d "C:\Simulation arena\backend"

echo.
echo   Simulation Arena is starting...
echo   The app will open at http://localhost:8000
echo   (Keep this window open. Close it to stop the app.)
echo.

start "" http://localhost:8000
"%PY%" -m app.main
pause
