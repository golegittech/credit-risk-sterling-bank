@echo off
title Sterling Bank Credit Risk Decision Engine
echo ==============================================================
echo  STERLING BANK CREDIT RISK ANALYSIS & DECISION ENGINE
echo  Student: Mbaegbu Chinedum Grace (2025/A/DSC/0058)
echo ==============================================================
echo.
set PATH=C:\Users\hp\AppData\Local\Programs\Python\Python311;C:\Users\hp\AppData\Local\Programs\Python\Python311\Scripts;%PATH%
set PYTHONIOENCODING=utf-8

echo [1/2] Starting FastAPI Backend on http://localhost:8000...
start "Sterling API Backend" /min cmd /k "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000"

echo [2/2] Starting Streamlit Underwriting Dashboard on http://localhost:8501...
start "Sterling Dashboard" cmd /k "python -m streamlit run app/streamlit_app.py --server.port 8501 --server.address localhost"

echo.
echo Launching your browser to http://localhost:8501...
timeout /t 3 >nul
start http://localhost:8501

echo.
echo The application is now running!
echo - Underwriting Dashboard: http://localhost:8501
echo - API Documentation: http://localhost:8000/docs
echo.
pause
