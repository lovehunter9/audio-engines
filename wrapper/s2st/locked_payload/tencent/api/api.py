"""One composite engine API, with fixed-profile file and live transports."""
import asyncio
import base64
import concurrent.futures
from contextlib import asynccontextmanager
import io
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import wave

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from anyio import CancelScope
from wrapper import tasks

from .file_session import FileSession
from .lifecycle import services
from .live_session import LiveSession


def resolve_direction(direction=None, source_lang=None, target_lang=None):
    if direction:
        if direction not in ('zh2en', 'en2zh'):
            raise HTTPException(400, 'direction must be zh2en or en2zh')
        return direction
    aliases = {'en': 'en', 'english': 'en', 'eng': 'en', 'en-us': 'en', 'en-gb': 'en',
               'zh': 'zh', 'chinese': 'zh', 'zh-cn': 'zh', 'zh-tw': 'zh', 'cn': 'zh', 'zhs': 'zh', 'zht': 'zh'}
    source = aliases.get(str(source_lang or 'English').lower())
    target = aliases.get(str(target_lang or 'Chinese').lower())
    if (source, target) not in [('zh', 'en'), ('en', 'zh')]:
        raise HTTPException(400, 'source_lang and target_lang must be Chinese and English in opposite directions')
    return source + '2' + target


def format_payload(payload, response_format):
    if response_format == 'json':
        return payload
    if response_format == 'text':
        return tasks.Binary(payload['translation'].encode(), 'text/plain', suffix='.txt')
    raw = base64.b64decode(payload['audio_base64'], validate=True)
    if response_format == 'wav':
        return tasks.Binary(raw, 'audio/wav', suffix='.wav')
    with wave.open(io.BytesIO(raw), 'rb') as reader:
        raw = reader.readframes(reader.getnframes())
    return tasks.Binary(raw, 'audio/L16', suffix='.pcm')


def build_app(family, args, supports):
    device = args.text('--device', 'cuda')
    if device != 'cuda':
        raise ValueError('This frozen image supports CUDA only; Intel adapter must use OpenVINO')
    role = args.text('--task-role', 's2st')
    if role not in ('s2st', 's2t'):
        raise ValueError('Unknown fixed task role')
    cache = Path(args.text('--cache-dir', '/data/s2st'))
    retain_measurements = args.switch('--retain-measurements', False)
    models = {
        'translation': args.text('--translation-model'),
        'translation_meta': args.text('--translation-metadata'),
        'speech': args.text('--speech-model'),
        'vad': args.text('--vad-model'),
    }
    if family == 'youdao':
        models.update(recognition=args.text('--recognition-model'))
    args.warn_unclaimed(logging.getLogger('audio-s2st'))
    state = {'ready': False, 'fault': None, 'context': None}
    active = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app):
        async def boot():
            try:
                if not all(models.values()):
                    raise ValueError('Pass every local model asset path through ENGINE_ARGS; serving never downloads model weights')
                async with services(family, cache, models, role) as context:
                    state.update(ready=True, context=context)
                    await asyncio.Future()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                state.update(ready=False, fault=str(error))
                logging.getLogger('audio-s2st').exception('Locked engine startup failed')
        starting = asyncio.create_task(boot())
        try:
            yield
        finally:
            starting.cancel()
            try:
                await starting
            except asyncio.CancelledError:
                pass
            finally:
                state['ready'] = False

    app = FastAPI(lifespan=lifespan)
    app.state.locked_status = state

    def is_ready():
        if not state['ready'] or state['fault']:
            return False
        return state['context'].get('is_alive', lambda: True)()
    app.state.locked_is_ready = is_ready

    def available():
        if not is_ready():
            raise HTTPException(503, 'Locked engine is not ready')
        if active.locked():
            raise HTTPException(409, 'This fixed profile supports one active translation task')

    async def invoke(session, emit, queued=False):
        if not queued:
            available()
        async with active:
            if not is_ready():
                raise HTTPException(503, 'Locked engine is not ready')
            running = asyncio.create_task(session.run())
            try:
                async for event in session.events(running):
                    await emit(event)
                result = await running
            except BaseException as error:
                # An interrupted original backend request may retain session
                state['fault'] = type(error).__name__
                raise
            finally:
                with CancelScope(shield=True):
                    running.cancel()
                    await asyncio.gather(running, return_exceptions=True)
        await emit({'type': 'done', 'translation': result.get('translation', result.get('translated_text')),
                    'source_samples': result.get('source_samples', result.get('source_samples_sent')),
                    'result': result})
        return result

    def release(session, request_root=None):
        if request_root:
            shutil.rmtree(request_root, ignore_errors=True)
        if not retain_measurements:
            shutil.rmtree(session.directory, ignore_errors=True)

    async def complete_payload(session, queued=False):
        async def discard(event):
            pass
        result = await invoke(session, discard, queued=queued)
        terminal = {'type': 'done', 'translation': result.get('translation', result.get('translated_text')),
                    'source_samples': result.get('source_samples', result.get('source_samples_sent')),
                    'result': result}
        terminal['text'] = terminal['translation']
        if role == 's2st':
            terminal.update(sample_rate=24000, audio_base64=base64.b64encode(
                (session.directory / 'ming/translated.wav').read_bytes()).decode())
            terminal['audio'] = terminal['audio_base64']
        return terminal

    @app.post('/v1/audio/translations')
    async def translate_file(file: UploadFile = File(...), direction: str = Form(None),
                             input_mode: str = Form('asap'), stream: bool = Form(False),
                             async_: str = Query(None, alias='async'),
                             source_lang: str = Form(None), target_lang: str = Form(None),
                             response_format: str = Form('json')):
        direction = resolve_direction(direction, source_lang, target_lang)
        fmt = str(response_format).lower()
        if fmt not in ('json', 'text', 'wav', 'pcm'):
            raise HTTPException(400, 'response_format must be json, text, wav, or pcm')
        if (stream or role == 's2t') and fmt != 'json':
            raise HTTPException(400, 'Use response_format=json for streaming or standalone text mode')
        asynchronous = tasks.truthy(async_)
        if asynchronous:
            if stream:
                raise HTTPException(400, 'Use stream=false with async=1; fetch the completed task result')
            if not is_ready():
                raise HTTPException(503, 'Locked engine is not ready')
        else:
            available()
        request_root = Path(tempfile.mkdtemp(prefix='upload-', dir=cache))
        source = request_root / 'source.wav'
        try:
            with source.open('wb') as writer:
                while chunk := await file.read(1024 * 1024):
                    writer.write(chunk)
            session = FileSession(state['context'], family, source, direction, input_mode)
        except (ValueError, EOFError, wave.Error) as error:
            shutil.rmtree(request_root)
            raise HTTPException(400, str(error)) from error
        finally:
            await file.close()
        if asynchronous:
            loop = asyncio.get_running_loop()
            def work(ctx):
                ctx.checkpoint()
                ctx.progress(ratio=0, stage='translation')
                future = asyncio.run_coroutine_threadsafe(complete_payload(session, queued=True), loop)
                while True:
                    try:
                        payload = future.result(timeout=.1)
                        break
                    except concurrent.futures.TimeoutError:
                        if future.done():
                            raise
                        # A running original request drains before cancellation
                        continue
                ctx.checkpoint()
                ctx.meter(input_seconds=session.source_samples / 16000)
                if role == 's2st':
                    with wave.open(str(session.directory / 'ming/translated.wav'), 'rb') as reader:
                        ctx.meter(output_seconds=reader.getnframes() / reader.getframerate())
                ctx.progress(ratio=1, stage='done')
                return format_payload(payload, fmt)
            return await tasks.dispatch(async_, 'audio_s2s', os.environ.get('MODEL_NAME', family),
                                        work, cleanup=lambda: release(session, request_root),
                                        fail='speech translation failed')
        if stream:
            queue = asyncio.Queue(maxsize=8)
            async def produce():
                try:
                    await invoke(session, queue.put)
                except Exception as error:
                    await queue.put({'type': 'error', 'message': str(error)})
                finally:
                    if not asyncio.current_task().cancelling():
                        await queue.put(None)
            async def body():
                worker = asyncio.create_task(produce())
                try:
                    while (event := await queue.get()) is not None:
                        yield json.dumps(event, ensure_ascii=False) + '\n'
                    await worker
                finally:
                    with CancelScope(shield=True):
                        worker.cancel()
                        await asyncio.gather(worker, return_exceptions=True)
                        release(session, request_root)
            return StreamingResponse(body(), media_type='application/x-ndjson')
        async def complete_response():
            try:
                payload = format_payload(await complete_payload(session), fmt)
                yield json.dumps(payload, ensure_ascii=False) if fmt == 'json' else payload.data
            except Exception as error:
                yield json.dumps({'type': 'error', 'message': str(error)})
            finally:
                release(session, request_root)
        # Headers precede long model work, preserving the existing non-stream
        media = {'json': 'application/json', 'text': 'text/plain', 'wav': 'audio/wav', 'pcm': 'audio/L16'}[fmt]
        return StreamingResponse(complete_response(), media_type=media)

    @app.websocket('/v1/audio/translations/stream')
    async def translate_live(socket: WebSocket):
        await socket.accept()
        worker = receiving = session = None
        try:
            available()
            init = await asyncio.wait_for(socket.receive_json(), 30)
            init['direction'] = resolve_direction(init.get('direction'), init.get('source_lang'), init.get('target_lang'))
            if init.get('sample_rate', 16000) != 16000 or init.get('format', 'pcm_s16le') != 'pcm_s16le':
                raise ValueError('Live input requires mono PCM16 at16000Hz')
            session = LiveSession(state['context'], family, init['direction'])
            worker = asyncio.create_task(invoke(session, socket.send_json))
            await socket.send_json({'type': 'ready', 'sample_rate': 16000, 'format': 'pcm_s16le',
                                    'source_tail': 'Caller supplies its exact source; no hidden silence/padding'})
            while True:
                receiving = asyncio.create_task(socket.receive())
                done, _ = await asyncio.wait([receiving, worker], return_when=asyncio.FIRST_COMPLETED)
                if worker in done:
                    await worker
                    break
                packet = receiving.result()
                if packet['type'] == 'websocket.disconnect':
                    raise WebSocketDisconnect()
                if packet.get('bytes') is not None:
                    await session.feed(packet['bytes'])
                elif json.loads(packet.get('text') or '{}') == {'type': 'end'}:
                    await session.end()
                    await worker
                    break
                else:
                    raise ValueError('Expected PCM16 bytes or a single end record')
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception as error:
            try:
                await socket.send_json({'type': 'error', 'message': str(error)})
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            with CancelScope(shield=True):
                for task in (worker, receiving):
                    if task:
                        task.cancel()
                await asyncio.gather(*(task for task in (worker, receiving) if task), return_exceptions=True)
                if session:
                    release(session)
                try:
                    await socket.close()
                except RuntimeError:
                    pass

    return app
