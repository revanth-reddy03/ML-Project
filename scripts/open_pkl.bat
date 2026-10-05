@echo off
set "PROJECT_ROOT=%~dp0.."
"%PROJECT_ROOT%\.venv\Scripts\python.exe" "%~dp0open_pickle.py" "%~1"
if errorlevel 1 pause