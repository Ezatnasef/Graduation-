@echo off
REM Start backend with reload disabled
cd /d "d:\Desktop\programing\CSAP\Servia_Voice"
set SERVIA_RELOAD=0
set SERVIA_PORT=8765
python -u backend/main.py
