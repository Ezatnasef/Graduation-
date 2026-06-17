import sys, pathlib, asyncio, time
sys.path.insert(0, str(pathlib.Path('backend').resolve()))
from main import _get_or_create_api_session, _make_voice_pipeline_for_session
import tts_engine, stt_engine

sample = str(pathlib.Path('models') / 'sample_06.wav')
with open(sample,'rb') as f:
    audio = f.read()

async def run_test(n=10):
    session = _get_or_create_api_session('loadtest','cairene','female')
    pipeline = _make_voice_pipeline_for_session(session)
    for i in range(n):
        print(f'=== Run {i+1} ===')
        before_vram = stt_engine._get_vram_usage_percent() if hasattr(stt_engine,'_get_vram_usage_percent') else 0.0
        start = time.perf_counter()
        out = await pipeline.run_audio_turn(audio_bytes=audio, mime_type='audio/wav', tts_dialect='cairene', tts_gender='female', language='ar', include_tts=False)
        duration = (time.perf_counter() - start)*1000
        after_vram = stt_engine._get_vram_usage_percent() if hasattr(stt_engine,'_get_vram_usage_percent') else 0.0
        print('Total latency_ms:', out.get('latency_ms'), 'measured_ms:', int(duration))
        stt = out.get('stt')
        print('STT provider, confidence, latency:', stt.get('provider'), stt.get('confidence'), stt.get('latency_ms'))
        print('VRAM before=%.1f%% after=%.1f%%' % (before_vram, after_vram))
        await asyncio.sleep(0.3)
    return

if __name__ == '__main__':
    asyncio.run(run_test(10))
