import asyncio
import json
import re
import time


_CONTROL_PREFIX = re.compile(
    r"^<?\s*(?:W(?:A(?:I(?:T)?)?)?|T(?:R(?:A(?:N(?:S)?)?)?)?)?\s*>?\s*[:：]?\s*$", re.I
)


class TargetText:
    def __init__(self, direction, emit, join, parse):
        self.direction = direction
        self.emit = emit
        self.join = join
        self.parse = parse
        self.parts = []
        self.raw = ""
        self.delivered = ""
        self.sequence = 0
        self.boundary_space = False

    def begin(self):
        if self.raw:
            raise RuntimeError("Previous translation completion did not finish")
        self.sequence += 1

    async def update(self, delta):
        self.raw += delta
        if _CONTROL_PREFIX.fullmatch(self.raw.strip()):
            return
        _, text = self.parse(self.raw)
        value = self.join([*self.parts, text], self.direction)
        if self.direction == "zh2en":
            words = list(re.finditer(r"\S+", value))
            safe = value[: words[-2].start()] if len(words) >= 3 else ""
        else:
            safe = value.rstrip()
        if len(safe) > len(self.delivered):
            await self.publish(safe)

    async def publish(self, value):
        if not value.startswith(self.delivered):
            raise RuntimeError("Translation attempted to revise already delivered text")
        delta = value[len(self.delivered) :]
        if delta:
            wire = delta[1:] if self.boundary_space and delta.startswith(" ") else delta
            self.boundary_space = False
            await self.emit(
                dict(
                    type="translation_delta",
                    sequence=self.sequence,
                    text=wire,
                    canonical_text=delta,
                )
            )
            self.delivered = value

    async def finish(self):
        raw, self.raw = self.raw, ""
        _, text = self.parse(raw)
        if not text:
            return raw
        value = self.join([*self.parts, text], self.direction)
        await self.publish(value)
        self.parts.append(text)
        if self.direction == "zh2en" and not self.boundary_space:
            await self.emit(
                dict(
                    type="translation_delta",
                    sequence=self.sequence,
                    text=" ",
                    canonical_text="",
                    word_boundary=True,
                )
            )
            self.boundary_space = True
        await self.emit(dict(type="translation_commit", sequence=self.sequence, text=text))
        return raw


class CompletionStream:
    def __init__(self, native, direction, target, stage=None, observe=None, client=None):
        self.native = native
        self.direction = direction
        self.target = target
        self.stage = stage
        self.observe = observe
        self.client = client
        self.lock = asyncio.Lock()

    def payload(self, message, force):
        native = self.native
        body = dict(
            model=native.model,
            temperature=0.0,
            max_tokens=native.force_max_new_tokens if force else native.max_new_tokens,
            messages=[
                dict(role="system", content="You are a helpful assistant."),
                dict(role="user", content=message),
            ],
            stream=True,
        )
        if force and native.min_tokens_on_force:
            if native.min_tokens_on_force != 1:
                raise ValueError("Unsupported forced minimum token count")
            body["grammar"] = r"root ::= [^\x00]+"
        if not force:
            if native.logit_bias:
                body["logit_bias"] = {str(key): value for key, value in native.logit_bias.items()}
            if native.repetition_penalty is not None:
                body["repeat_penalty"] = native.repetition_penalty
        return body

    async def __call__(self, message, force=False):
        async with self.lock:
            if self.client is None:
                import httpx

                self.client = httpx.AsyncClient(
                    timeout=self.native.timeout_seconds, trust_env=False
                )
            if self.stage is None:
                return await self.generate(message, force)
            async with self.stage("t3po"):
                return await self.generate(message, force)

    async def generate(self, message, force):
        started = time.monotonic()
        self.target.begin()
        body = self.payload(message, force)
        headers = {"content-type": "application/json"}
        if self.native.api_key:
            headers["authorization"] = "Bearer " + self.native.api_key
        if self.observe:
            await self.observe(
                dict(
                    event="translation_request",
                    sequence=self.target.sequence,
                    payload=body,
                    force=force,
                    started_monotonic_seconds=started,
                )
            )
        finish_reason = None
        done = False
        async with self.client.stream(
            "POST", self.native.url, json=body, headers=headers
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                value = line[5:].strip()
                if value == "[DONE]":
                    done = True
                    break
                item = json.loads(value)
                if item.get("error"):
                    raise RuntimeError("Translation endpoint returned an error")
                choices = item.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                text = choice.get("delta", {}).get("content") or ""
                if self.direction == "zh2en" and re.search(r"[\u3400-\u9fff\uf900-\ufaff]", text):
                    raise RuntimeError(
                        "English-target translation produced Han characters; stream cannot be retracted"
                    )
                if text:
                    await self.target.update(text)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
        if not done or finish_reason != "stop":
            raise RuntimeError("Translation stream did not finish normally: " + repr(finish_reason))
        raw = await self.target.finish()
        if self.observe:
            await self.observe(
                dict(
                    event="translation_response",
                    sequence=self.target.sequence,
                    raw=raw,
                    force=force,
                    started_monotonic_seconds=started,
                    finished_monotonic_seconds=time.monotonic(),
                    timing_scope="MT streaming request after GPU gate; includes local HTTP/dispatch, not GPU kernel time",
                    retries=0,
                )
            )
        return raw

    async def close(self):
        if self.client is not None:
            await self.client.aclose()
            self.client = None




def install(observer_path):
    from inference import server
    from inference.prompts import parse_model_response
    from inference.translation import join_translation_segments
    from .gpu_queue import async_stage
    from .youdao_word_tail import SpeechWordTail

    original_send = server.SpeechSession.send
    original_run = server.SpeechSession.run

    async def send(self, event):
        if event.get("type") == "init_ok" and not hasattr(self, "_target_completion_stream"):
            native = await server.model_store.get_translation(self.latency_mode)
            self._direct_speech_characters = 0
            self._speech_word_tail = SpeechWordTail(self.direction)

            async def emit(value):
                value = dict(
                    value,
                    request_source_samples=getattr(
                        self._translation.engine, "_source_samples", None
                    ),
                    observed_source_samples=getattr(self, "_asr_latest_source_samples", None),
                )
                await original_send(self, value)
                if value.get("type") == "translation_delta":
                    ready = self._speech_word_tail.feed(value["text"])
                    if ready:
                        self._direct_speech_characters += len(ready)
                        await original_send(
                            self,
                            dict(value, type="speech_text", text=ready, upstream_word_tail=True),
                        )

            async def observe(value):
                with observer_path.open("a") as log:
                    log.write(
                        json.dumps(
                            dict(
                                value, direction=self.direction, monotonic_seconds=time.monotonic()
                            ),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            target = TargetText(
                self.direction, emit, join_translation_segments, parse_model_response
            )
            completion = CompletionStream(
                native, self.direction, target, stage=async_stage, observe=observe
            )
            self._target_completion_stream = completion
            self._translation.engine._complete = completion
        if event.get("type") == "ended" and hasattr(self, "_target_completion_stream"):
            tail = self._speech_word_tail.feed("", final=True)
            if tail:
                self._direct_speech_characters += len(tail)
                await original_send(
                    self,
                    dict(
                        type="speech_text",
                        text=tail,
                        request_source_samples=getattr(
                            self._translation.engine, "_source_samples", None
                        ),
                        observed_source_samples=getattr(self, "_asr_latest_source_samples", None),
                        upstream_word_tail=True,
                        final=True,
                    ),
                )
            await original_send(
                self,
                dict(
                    type="speech_text_integrity",
                    identical=True,
                    received_characters=self._direct_speech_characters,
                    adapter="lossless upstream open-word boundary",
                ),
            )
        return await original_send(self, event)

    async def run(self):
        try:
            return await original_run(self)
        finally:
            completion = getattr(self, "_target_completion_stream", None)
            if completion is not None:
                await completion.close()

    server.SpeechSession.send = send
    server.SpeechSession.run = run
