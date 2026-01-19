@echo off
echo Starting Medical RAG Frontend...
cd /d %~dp0frontend

if not exist node_modules (
    echo Installing dependencies...
    npm install
)

echo.
echo Starting dev server on http://localhost:3000
echo Press Ctrl+C to stop
echo.
npm run dev
