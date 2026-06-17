# Servia

Real-time Egyptian Arabic voice assistant with local LLM + advanced TTS + VAD/barge-in.

## New Production Enhancements

- Backend STT pipeline with provider fallback:
	- EgypTalk-ASR-v2 (local NeMo)
	- Whisper API (remote)
	- Faster-Whisper (local)
	- Browser STT compatibility fallback
- Conversation memory compression:
	- rolling short-term memory
	- LLM-based summarization for long conversations
- Modular pipeline orchestration endpoint:
	- Input -> STT -> Processing -> LLM -> TTS
- Evaluation toolkit for quality and latency checks
- Windows operational scripts (`scripts/run_dev.bat`, `scripts/run_prod.bat`, `scripts/batch_test.bat`, `tools/check_api.py`)

## Backend Endpoints

- `GET /` health
- `GET /api/dialects`
- `GET /api/tts/status`
- `GET /api/stt/status`
- `POST /api/tts`
- `POST /api/tts/audio`
- `POST /stt` or `POST /api/stt` (single upload STT)
- `POST /api/stt/chunk` (chunked STT upload)
- `POST /api/chat` (HTTP chat with session memory)
- `POST /api/session/clear`
- `POST /api/pipeline/voice-turn` (full modular voice turn)
- `GET /api/analytics/summary`
- `WS /ws/voice`

## Quick Start (Windows)

1. Create and activate venv.
2. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

3. Run in dev mode:

```bat
scripts\run_dev.bat
```

4. Run quick API checks:

```bat
scripts\batch_test.bat
```

## STT Environment Variables

- `STT_PROVIDER=auto|egyptalk|whisper_api|faster_whisper|browser`
- `STT_EGYPTALK_MODEL_ID` (default `NAMAA-Space/EgypTalk-ASR-v2`)
- `STT_EGYPTALK_LOCAL_MODEL` (optional local `.nemo` file or folder containing `.nemo`)
- `STT_EGYPTALK_TIMEOUT_SECONDS` (default `20`)
- `STT_WHISPER_API_KEY` (or `OPENAI_API_KEY`)
- `STT_WHISPER_API_URL` (default OpenAI transcription endpoint)
- `STT_WHISPER_MODEL` (default `gpt-4o-mini-transcribe`)
- `STT_FASTER_WHISPER_MODEL` (default `small`)
- `STT_FASTER_WHISPER_DEVICE` (default `auto`)
- `STT_FASTER_WHISPER_COMPUTE_TYPE` (default `int8`)
- `STT_MAX_LATENCY_MS` (default `7000`)
- `STT_MIN_CONFIDENCE` (default `0.35`)

## EgypTalk-ASR-v2 Setup (Windows)

1. Install `git-xet` (for large Hugging Face repos):

```powershell
winget install git-xet
```

2. Clone model repo locally (inside project root, optional):

```powershell
git clone https://huggingface.co/NAMAA-Space/EgypTalk-ASR-v2 models/EgypTalk-ASR-v2
```

3. Install backend dependencies (includes NeMo ASR):

```powershell
pip install -r backend/requirements.txt
```

4. Configure backend env:

```env
STT_PROVIDER=egyptalk
STT_EGYPTALK_LOCAL_MODEL=../models/EgypTalk-ASR-v2
```

If `STT_EGYPTALK_LOCAL_MODEL` is empty, backend loads `STT_EGYPTALK_MODEL_ID` from Hugging Face.

## Memory Summarization Variables

- `MEMORY_MAX_RECENT_MESSAGES` (default `12`)
- `MEMORY_SUMMARIZE_AFTER_MESSAGES` (default `16`)
- `MEMORY_SUMMARIZE_AFTER_TOKENS` (default `900`)

## Evaluation

- Sample dataset: `backend/evaluation/sample_eval_dataset.json`
- Evaluator script: `backend/evaluation/evaluate_pipeline.py`

Use it against running backend:

```bash
python backend/evaluation/evaluate_pipeline.py --base-url http://127.0.0.1:8765
```

## Docker Compose

Run from `deploy/`:

```bash
cd deploy
docker compose --env-file .env.docker.example up --build
```
