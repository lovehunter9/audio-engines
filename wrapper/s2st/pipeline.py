import asyncio
import base64
import hashlib
import time

from .speech import install, settings


async def run(source, producer, emit, speech_url, *, streaming=False):
    import numpy as np
    from simulinfer.tts import Speech

    install()
    config = settings(speech_url)
    target = ""
    spoken_target = ""
    separate_speech = getattr(producer, "separate_speech_events", False)
    published = 0
    done = False
    started = time.monotonic()
    delivered = 0
    queue = asyncio.Queue(maxsize=8)

    async def enqueue(item):
        queued = asyncio.create_task(queue.put(item))
        done, _ = await asyncio.wait({queued, sending}, return_when=asyncio.FIRST_COMPLETED)
        if queued in done:
            await queued
            return
        if sending in done:
            queued.cancel()
            await asyncio.gather(queued, return_exceptions=True)
            sending.result()
            raise RuntimeError("Audio publisher stopped before delivery finished")
        await queued

    async def publish(pcm, row):
        await enqueue((pcm, row))

    async def sender():
        nonlocal delivered
        first = True
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                pcm, row = item
                if first:
                    await asyncio.sleep(max(0.0, started + row["start_seconds"] - time.monotonic()))
                    first = False
                await emit(
                    {
                        "type": "audio",
                        "pcm_base64": base64.b64encode(pcm).decode(),
                        "sample_rate": 24000,
                        "channels": 1,
                        "format": "pcm_s16le",
                        "sequence": row["seq"],
                        "playout_speed": row["speed"],
                        "initial_buffer_applied": True,
                    }
                )
                delivered += len(pcm)
            finally:
                queue.task_done()

    player = None
    sending = None
    if streaming:
        from .playout import OnlinePlayout

        chinese = source.stream.direction == "en2zh"
        player = OnlinePlayout(
            producer.directory.parent,
            config.tts_speed if chinese else config.tts_speed_zh_en,
            prebuffer=2.0 if chinese else 1.0,
            target_queue=5.0 if chinese else 4.0,
            min_speed=0.95 if chinese else 1.0,
            publish=publish,
        )
        sending = asyncio.create_task(sender())

    async def speech_event(event):
        nonlocal published, done, delivered
        if event.get("event") == "audio":
            published += len(base64.b64decode(event["data"]))
            if player is not None:
                if sending.done():
                    sending.result()
                player.audio(event["seq"], base64.b64decode(event["data"]))
            else:
                await emit(
                    {
                        "type": "audio",
                        "pcm_base64": event["data"],
                        "sample_rate": 24000,
                        "channels": 1,
                        "format": "pcm_s16le",
                        "sequence": event["seq"],
                    }
                )
                delivered += len(base64.b64decode(event["data"]))
        elif event.get("event") == "tts_end" and player is not None:
            await player.complete(event["seq"], started)
        elif event.get("event") == "tts_error":
            raise RuntimeError("A complete MING phrase failed synthesis")
        elif event.get("event") == "tts_done":
            done = event["complete"]
        elif event.get("event") in ("tts_text_integrity", "tts_clause_integrity"):
            if not event["identical"]:
                raise RuntimeError("Shared MING text integrity check failed")

    direction = "zh-en" if source.stream.direction == "zh2en" else "en-zh"
    speech = Speech(config, speech_event, direction)
    if hasattr(producer, "speech_stage"):
        original_synthesize = speech._synthesize

        async def scheduled_synthesize(text):
            async with producer.speech_stage():
                return await original_synthesize(text)

        speech._synthesize = scheduled_synthesize

    async def source_consumed(pcm):
        if pcm:
            speech.feed_source(np.frombuffer(pcm, "<i2").astype(np.float32) / 32768)

    try:
        await speech.start()
        async for event in producer.run(source, source_consumed):
            if event["type"] == "translation_delta":
                target += event["text"]
                if not separate_speech:
                    spoken_target += event["text"]
                    await speech.text(event["text"])
            elif event["type"] == "speech_text":
                if not separate_speech:
                    raise RuntimeError("Unexpected separate speech text")
                spoken_target += event["text"]
                await speech.text(event["text"])
                continue
            await emit(event)
        await speech.finish()
        playout = None
        if player is not None:
            playout = player.finish(started, source.samples / 16000, published)
            if sending.done():
                sending.result()
            await enqueue(None)
            await sending
        state = getattr(speech, "_complete_words", None)
        if (
            not done
            or not published
            or state is None
            or state.received != state.committed
            or state.received != spoken_target
        ):
            raise RuntimeError("S2ST did not deliver all translated text and audio")
        await emit(
            {
                "type": "done",
                "translation": target,
                "source_samples": source.samples,
                "source_pcm_sha256": source.digest.hexdigest(),
                "translated_text_sha256": hashlib.sha256(target.encode()).hexdigest(),
                "output_samples": delivered // 2,
                "raw_tts_output_samples": published // 2,
                "online_playout": playout,
                "elapsed_seconds": time.monotonic() - started,
                "tts_attempts_per_phrase": 1,
                "tts_retries": 0,
                **getattr(producer, "completion_metadata", {}),
            }
        )
    finally:
        if sending is not None and not sending.done():
            sending.cancel()
            await asyncio.gather(sending, return_exceptions=True)
        if player is not None:
            player.writer.close()
            if not player.log.closed:
                player.log.close()
        await speech.close()
        await producer.close()
