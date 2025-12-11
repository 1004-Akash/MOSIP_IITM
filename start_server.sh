#!/bin/bash

echo "========================================"
echo "Starting OCR API Server"
echo "========================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if dependencies are installed
python3 -c "import fastapi" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "Starting server on http://localhost:8000"
echo ""
echo "Access points:"
echo "  - Web UI: http://localhost:8000/ui"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Health Check: http://localhost:8000/health"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

