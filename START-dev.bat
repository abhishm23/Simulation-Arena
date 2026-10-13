@echo off
title Simulation Arena (developer mode)
set "PY=C:\Users\Abhishek Mishra\.workbuddy-ai\binaries\python\envs\simarena\Scripts\python.exe"
set "NODE=C:\Users\Abhishek Mishra\.workbuddy-ai\binaries\node\versions\22.22.2-2"
set "PATH=%NODE%;%PATH%"

start "Simulation Arena - API" cmd /k "cd /d "C:\Simulation arena\backend" && "%PY%" -m app.main"
timeout /t 4 /nobreak >nul
start "Simulation Arena - UI" cmd /k "cd /d "C:\Simulation arena\frontend" && npm run dev"
timeout /t 5 /nobreak >nul
start "" http://localhost:5173
echo Developer mode running: API on :8000, UI on :5173
