import os, time, asyncio, json, sys
sys.path.insert(0, r'd:/Desktop/programing/CSAP/Servia/backend')

async def bench(provider):
    os.environ['TTS_PROVIDER']=provider
    import importlib
    import tts_engine
    importlib.reload(tts_engine)
    txt='اهلا بيك في سيرفيا، ازاي اقدر اساعدك؟'
    t0=time.perf_counter()
    try:
        audio, fmt = await tts_engine.synthesize_speech(txt,'cairene','female','neutral')
        t1=time.perf_counter()
        audio2, fmt2 = await tts_engine.synthesize_speech(txt,'cairene','female','neutral')
        t2=time.perf_counter()
        return {
            'provider':provider,
            'ok':True,
            'fmt':fmt,
            'first_ms':int((t1-t0)*1000),
            'second_ms':int((t2-t1)*1000),
            'bytes':len(audio)
        }
    except Exception as e:
        return {'provider':provider,'ok':False,'error':str(e)}

async def main():
    out=[]
    for p in ['xtts','chatterbox']:
        out.append(await bench(p))
    print(json.dumps(out, ensure_ascii=False))

asyncio.run(main())
