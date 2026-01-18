@echo off
REM BTC Trading System v4.0 - Windows Launcher
REM Double-click this file to start the system

title BTC Trading System v4.0

cd /d "%~dp0"

echo.
echo ================================================================================
echo                  BTC Trading System v4.0 - Starting...
echo ================================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

REM Run the Python launcher
python start.py



