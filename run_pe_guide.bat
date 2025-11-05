@echo off
cd /d %~dp0
call venv\Scripts\activate
python pe_guide.py
pause
