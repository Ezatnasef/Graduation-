@echo off
setlocal EnableExtensions EnableDelayedExpansion
echo ==========================================
echo   Servia Voice Backend - Egyptian TTS
echo ==========================================
echo.

cd /d "%~dp0"

:: Prefer workspace virtual environment when available
set "PYTHON_EXE=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" set "PYTHON_EXE=python"

:: Runtime defaults (can be overridden from environment)
if "%OLLAMA_MODEL%"=="" set OLLAMA_MODEL=qwen2.5:1.5b-instruct-q8_0
if "%OLLAMA_MAX_TOKENS%"=="" set OLLAMA_MAX_TOKENS=110
if "%OLLAMA_GENERATE_TIMEOUT_SECONDS%"=="" set OLLAMA_GENERATE_TIMEOUT_SECONDS=18
if "%OLLAMA_MAX_RETRIES%"=="" set OLLAMA_MAX_RETRIES=1
if "%SERVIA_PORT%"=="" set SERVIA_PORT=8765
if "%TTS_PROVIDER%"=="" set TTS_PROVIDER=xtts
if "%TTS_AUTO_LOCAL_ORDER%"=="" set TTS_AUTO_LOCAL_ORDER=chatterbox,xtts
if "%TTS_AUTO_USE_DUAL%"=="" set TTS_AUTO_USE_DUAL=0
if "%TTS_HUMAN_STYLE_ENABLED%"=="" set TTS_HUMAN_STYLE_ENABLED=0
if "%XTTS_HALF_PRECISION%"=="" set XTTS_HALF_PRECISION=0
if "%XTTS_TIMEOUT_SECONDS%"=="" set XTTS_TIMEOUT_SECONDS=60
if "%XTTS_FAILURE_COOLDOWN_SECONDS%"=="" set XTTS_FAILURE_COOLDOWN_SECONDS=45
if "%CHATTERBOX_FAILURE_COOLDOWN_SECONDS%"=="" set CHATTERBOX_FAILURE_COOLDOWN_SECONDS=45
if "%TTS_WARMUP_TIMEOUT_SECONDS%"=="" set TTS_WARMUP_TIMEOUT_SECONDS=180
if "%XTTS_MODEL_DIR%"=="" set XTTS_MODEL_DIR=%~dp0..\models\NileTTS-XTTS
if "%CHATTERBOX_MODEL_DIR%"=="" set CHATTERBOX_MODEL_DIR=%~dp0..\models\chatterbox-egyptian-v0
if "%STT_EGYPTALK_LOCAL_MODEL%"=="" if exist "%~dp0..\models\EgypTalk-ASR-v2" set STT_EGYPTALK_LOCAL_MODEL=%~dp0..\models\EgypTalk-ASR-v2
if "%XTTS_REFERENCE_AUDIO%"=="" set XTTS_REFERENCE_AUDIO=%~dp0..\models\sample_06.wav
if "%SERVIA_SKIP_INSTALL%"=="" set SERVIA_SKIP_INSTALL=0
if "%KMP_DUPLICATE_LIB_OK%"=="" set KMP_DUPLICATE_LIB_OK=TRUE
if "%STT_PROVIDER%"=="" set STT_PROVIDER=egyptalk
if "%STT_AUTO_USE_EGYPTALK%"=="" set STT_AUTO_USE_EGYPTALK=1
if "%STT_EGYPTALK_TIMEOUT_SECONDS%"=="" set STT_EGYPTALK_TIMEOUT_SECONDS=30
if "%STT_FASTER_WHISPER_MODEL%"=="" set STT_FASTER_WHISPER_MODEL=small

set PORT_BUSY=
for /f "tokens=*" %%L in ('netstat -ano ^| findstr /R /C:":%SERVIA_PORT% .*LISTENING"') do set PORT_BUSY=1
if defined PORT_BUSY (
    if "%SERVIA_PORT%"=="8765" (
        echo [WARN] Port 8765 is busy. Switching to port 8877.
        set SERVIA_PORT=8877
    ) else (
        echo [ERROR] Port %SERVIA_PORT% is busy. Set SERVIA_PORT to a free port and retry.
        pause
        exit /b 1
    )
)

set "PIP_DISABLE_PIP_VERSION_CHECK=1"

:: Check Python
"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.9+ from https://www.python.org
    pause
    exit /b 1
)

:: Check if runtime deps are already available
set "DEPS_OK=0"
"%PYTHON_EXE%" -c "import fastapi,uvicorn,websockets,gtts,edge_tts,numpy,pydantic,multipart,aiofiles,torch,transformers,safetensors,TTS,chatterbox" >nul 2>&1
if not errorlevel 1 set "DEPS_OK=1"

if "%SERVIA_SKIP_INSTALL%"=="1" (
    echo Skipping dependency installation by SERVIA_SKIP_INSTALL=1
) else if "%DEPS_OK%"=="1" (
    echo Dependencies already satisfied. Skipping install.
) else (
    echo Installing dependencies...
    "%PYTHON_EXE%" -m pip install --prefer-binary -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies!
        echo Hint: ensure the virtual environment exists at ..\.venv and retry.
        pause
        exit /b 1
    )
)

for /f "usebackq delims=" %%i in (`"%PYTHON_EXE%" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2^>nul`) do set "FFMPEG_BINARY=%%i"
if defined FFMPEG_BINARY (
    for %%i in ("%FFMPEG_BINARY%") do set "FFMPEG_DIR=%%~dpi"
    set "PATH=!FFMPEG_DIR!;!PATH!"
)

echo.
echo Starting Servia Voice Server on http://localhost:%SERVIA_PORT%
echo WebSocket: ws://localhost:%SERVIA_PORT%/ws/voice
echo API Docs:  http://localhost:%SERVIA_PORT%/docs
echo Ollama model: %OLLAMA_MODEL%
echo Max tokens:   %OLLAMA_MAX_TOKENS%
echo Max retries:  %OLLAMA_MAX_RETRIES%
echo TTS provider:  %TTS_PROVIDER%
echo TTS local order: %TTS_AUTO_LOCAL_ORDER%
echo TTS auto dual: %TTS_AUTO_USE_DUAL%
echo Human style cues: %TTS_HUMAN_STYLE_ENABLED%
echo XTTS timeout:  %XTTS_TIMEOUT_SECONDS%s
echo XTTS half precision: %XTTS_HALF_PRECISION%
echo Warmup timeout: %TTS_WARMUP_TIMEOUT_SECONDS%s
echo XTTS model dir: %XTTS_MODEL_DIR%
echo Chatterbox dir: %CHATTERBOX_MODEL_DIR%
echo EgypTalk dir:  %STT_EGYPTALK_LOCAL_MODEL%
echo STT provider:  %STT_PROVIDER%
echo FFmpeg:        %FFMPEG_BINARY%
echo Python exe:    %PYTHON_EXE%
echo.
echo Press Ctrl+C to stop the server.
echo ==========================================
echo.

"%PYTHON_EXE%" main.py
pause
