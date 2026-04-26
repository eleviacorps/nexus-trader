@echo off
cd /d c:\PersonalDrive\Programming\AiStudio\nexus-trader
set PYTHONPATH=c:\PersonalDrive\Programming\AiStudio\nexus-trader
"C:\Users\rfsga\miniconda3\python.exe" -m uvicorn src.service.app_v21:app --host 127.0.0.1 --port 8021
