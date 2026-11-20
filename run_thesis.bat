
@echo off
title Thesis RAG Application

echo ========================================
echo       Starting Thesis RAG Application
echo ========================================

cd /d "C:\My Dev\Upgrad\Thesis\Thesis-Coding"

echo.
echo [1/4] Activating Python virtual environment...
call mydev\Scripts\activate.bat

echo.
echo [2/4] Setting environment variables...
set USE_STUB_LLM=true
set RAG_DECISION_MODE=extract_compare

echo.
echo [3/4] Rebuilding FAISS index...
python -m src.pipeline

if errorlevel 1 (
    echo ERROR: Pipeline execution failed!
    pause
    exit /b 1
)

echo.
echo [4/4] Starting FastAPI server...
echo.
echo API will be available at:
echo http://127.0.0.1:8000
echo.
echo Swagger documentation:
echo http://127.0.0.1:8000/docs
echo.
echo Press CTRL+C to stop the server.
echo ========================================

python -m uvicorn src.api:app --reload

pause
