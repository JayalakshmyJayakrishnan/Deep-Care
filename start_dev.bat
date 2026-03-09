@echo off
echo Starting PillCam Platform...

echo Starting Backend Server (Flask)...
start "PillCam Backend" cmd /k "cd backend && python app.py"

echo Starting Frontend Server (Vite)...
start "PillCam Frontend" cmd /k "cd frontend && npm run dev"

echo Done. Check the opened windows for server logs.
