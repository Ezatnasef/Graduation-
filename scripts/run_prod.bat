@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%\backend"
if exist "%PROJECT_ROOT%\.venv\Scripts\activate.bat" call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
set SERVIA_ENV=prod

if "%OLLAMA_BASE_URL%"=="" set OLLAMA_BASE_URL=http://127.0.0.1:11434
if "%OLLAMA_MODEL%"=="" set OLLAMA_MODEL=qwen2.5:1.5b-instruct-q8_0
if "%OLLAMA_MAX_TOKENS%"=="" set OLLAMA_MAX_TOKENS=140
if "%OLLAMA_GENERATE_TIMEOUT_SECONDS%"=="" set OLLAMA_GENERATE_TIMEOUT_SECONDS=18
if "%OLLAMA_MAX_RETRIES%"=="" set OLLAMA_MAX_RETRIES=1
if "%TTS_PROVIDER%"=="" set TTS_PROVIDER=xtts
if "%TTS_AUTO_LOCAL_ORDER%"=="" set TTS_AUTO_LOCAL_ORDER=chatterbox,xtts
if "%TTS_HUMAN_STYLE_ENABLED%"=="" set TTS_HUMAN_STYLE_ENABLED=0
if "%XTTS_HALF_PRECISION%"=="" set XTTS_HALF_PRECISION=0
if "%XTTS_TIMEOUT_SECONDS%"=="" set XTTS_TIMEOUT_SECONDS=60
if "%XTTS_FAILURE_COOLDOWN_SECONDS%"=="" set XTTS_FAILURE_COOLDOWN_SECONDS=45
if "%CHATTERBOX_FAILURE_COOLDOWN_SECONDS%"=="" set CHATTERBOX_FAILURE_COOLDOWN_SECONDS=45
if "%TTS_WARMUP_TIMEOUT_SECONDS%"=="" set TTS_WARMUP_TIMEOUT_SECONDS=180
if "%SERVIA_PORT%"=="" set SERVIA_PORT=8765
if "%KMP_DUPLICATE_LIB_OK%"=="" set KMP_DUPLICATE_LIB_OK=TRUE
if "%XTTS_MODEL_DIR%"=="" set XTTS_MODEL_DIR=%PROJECT_ROOT%\models\NileTTS-XTTS
if "%XTTS_REFERENCE_AUDIO%"=="" set XTTS_REFERENCE_AUDIO=%PROJECT_ROOT%\models\sample_06.wav
if "%STT_PROVIDER%"=="" set STT_PROVIDER=egyptalk
if "%STT_AUTO_USE_EGYPTALK%"=="" set STT_AUTO_USE_EGYPTALK=1
if "%STT_EGYPTALK_LOCAL_MODEL%"=="" if exist "%PROJECT_ROOT%\models\EgypTalk-ASR-v2" set STT_EGYPTALK_LOCAL_MODEL=%PROJECT_ROOT%\models\EgypTalk-ASR-v2
if "%STT_EGYPTALK_TIMEOUT_SECONDS%"=="" set STT_EGYPTALK_TIMEOUT_SECONDS=30
if "%STT_FASTER_WHISPER_MODEL%"=="" set STT_FASTER_WHISPER_MODEL=small

for /f "delims=" %%i in ('python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2^>nul') do set "FFMPEG_BINARY=%%i"
if defined FFMPEG_BINARY (
	for %%i in ("%FFMPEG_BINARY%") do set "FFMPEG_DIR=%%~dpi"
	set "PATH=%FFMPEG_DIR%;%PATH%"
)

set PORT_BUSY=
for /f "tokens=*" %%L in ('netstat -ano ^| findstr /R /C:":%SERVIA_PORT% .*LISTENING"') do set PORT_BUSY=1
if defined PORT_BUSY (
	if "%SERVIA_PORT%"=="8765" (
		echo [WARN] Port 8765 is busy. Switching to port 8877.
		set SERVIA_PORT=8877
	) else (
		echo [ERROR] Port %SERVIA_PORT% is busy. Set SERVIA_PORT to a free port and retry.
		exit /b 1
	)
)

uvicorn main:app --host 0.0.0.0 --port %SERVIA_PORT% --workers 1
