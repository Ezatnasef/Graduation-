@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat
python tools\check_api.py
python backend\evaluation\evaluate_pipeline.py --base-url http://127.0.0.1:8765
