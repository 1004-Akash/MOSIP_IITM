@echo off
echo ========================================
echo Starting OCR API Server
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if dependencies are installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo Starting server on http://localhost:8000
echo.
echo Access points:
echo   - Web UI: http://localhost:8000/ui
echo   - API Docs: http://localhost:8000/docs
echo   - Health Check: http://localhost:8000/health
echo.
echo Press CTRL+C to stop the server
echo.

python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause

