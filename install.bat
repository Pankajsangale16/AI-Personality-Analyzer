@echo off
echo ========================================
echo AI Personality Prediction System
echo Installation Script
echo ========================================
echo.

echo Checking Python installation...
py --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found!
py --version

echo.
echo Checking pip...
py -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not installed
    pause
    exit /b 1
)

echo pip found!
py -m pip --version

echo.
echo Installing project dependencies...
py -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo Setting up Django project...
py setup.py

if %errorlevel% neq 0 (
    echo ERROR: Failed to setup Django project
    pause
    exit /b 1
)

echo.
echo Training ML models...
py train_models.py

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo To run application:
echo   py manage.py runserver
echo.
echo Then open: http://127.0.0.1:8000
echo.
pause
