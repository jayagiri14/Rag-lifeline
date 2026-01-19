@echo off
echo Starting Medical RAG Backend...
cd /d %~dp0backend

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting server on http://localhost:8000
echo Press Ctrl+C to stop
echo.
python -m uvicorn app.main:app --reload --port 8000
