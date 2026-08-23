@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE=.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

"%PYTHON_EXE%" interactive_demo.py --help >nul 2>&1
if errorlevel 1 (
  echo [ERROR] The XDreamer Python environment is not ready.
  echo.
  echo With Python 3.11 selected, run from this folder:
  echo   python -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install torch
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

"%PYTHON_EXE%" interactive_demo.py %*
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%

