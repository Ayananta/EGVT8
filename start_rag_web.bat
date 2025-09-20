@echo off
echo ========================================
echo    Elita Garden Vista Tower 8 Case
echo           RAG Web Interface
echo ========================================
echo.
echo Starting the web application...
echo.

REM Activate virtual environment and start the web app
call venv\Scripts\activate.bat
python start_web_app.py

pause
