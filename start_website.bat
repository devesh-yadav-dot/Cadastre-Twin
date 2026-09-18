@echo off
title SIH26011 Website Server

cd /d C:\Users\DEVESH\Desktop\sih\sih2

echo Starting SIH26011 Streamlit app...
start "SIH26011 Streamlit" powershell -NoExit -ExecutionPolicy Bypass -Command ".\cadastre_env\Scripts\Activate.ps1; streamlit run app.py"

timeout /t 5 /nobreak >nul

echo Starting Cloudflare Tunnel...
start "Cloudflare Tunnel" powershell -NoExit -ExecutionPolicy Bypass -Command "& 'C:\Program Files (x86)\cloudflared\cloudflared.exe' tunnel run mechrono-tunnel"

echo.
echo ==========================================
echo   SIH26011 WEBSITE IS STARTING
echo   https://mechrono.online
echo ==========================================
echo.
pause