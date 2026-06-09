@echo off
echo Starting CottonGuard Server...

echo The app will run on: http://localhost:8000
echo.
start "CottonGuard App" cmd /k "python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"

echo Server started in a new window! 
echo Please open your browser and navigate to: http://localhost:8000
echo.
pause
