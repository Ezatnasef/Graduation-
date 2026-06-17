@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%\backend"
if exist "%PROJECT_ROOT%\.venv\Scripts\activate.bat" call "%PROJECT_ROOT%\.venv\Scripts\activate.bat"
set SERVIA_ENV=dev
if "%KMP_DUPLICATE_LIB_OK%"=="" set KMP_DUPLICATE_LIB_OK=TRUE
if "%OLLAMA_MODEL%"=="" set OLLAMA_MODEL=qwen2.5:1.5b-instruct-q8_0
if "%XTTS_MODEL_DIR%"=="" set XTTS_MODEL_DIR=%PROJECT_ROOT%\models\NileTTS-XTTS
if "%XTTS_REFERENCE_AUDIO%"=="" set XTTS_REFERENCE_AUDIO=%PROJECT_ROOT%\models\sample_06.wav
if "%XTTS_HALF_PRECISION%"=="" set XTTS_HALF_PRECISION=0
if "%STT_PROVIDER%"=="" set STT_PROVIDER=egyptalk
if "%STT_AUTO_USE_EGYPTALK%"=="" set STT_AUTO_USE_EGYPTALK=1
if "%STT_EGYPTALK_LOCAL_MODEL%"=="" if exist "%PROJECT_ROOT%\models\EgypTalk-ASR-v2" set STT_EGYPTALK_LOCAL_MODEL=%PROJECT_ROOT%\models\EgypTalk-ASR-v2
if "%STT_EGYPTALK_TIMEOUT_SECONDS%"=="" set STT_EGYPTALK_TIMEOUT_SECONDS=30

for /f "delims=" %%i in ('python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2^>nul') do set "FFMPEG_BINARY=%%i"
if defined FFMPEG_BINARY (
	for %%i in ("%FFMPEG_BINARY%") do set "FFMPEG_DIR=%%~dpi"
	set "PATH=%FFMPEG_DIR%;%PATH%"
)
python main.py
