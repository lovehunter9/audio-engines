import numpy as np

STREAM_FRAMES = 8


class SilenceTail:
    def __init__(self, sr):
        self.frame = max(1, int(sr * 0.1))
        self.tail = max(1, int(sr * 0.3))
        self.pending = np.zeros(0, dtype="float32")
        self.voiced = False

    def push(self, wave, final=False):
        wave = np.asarray(wave, dtype="float32").reshape(-1)
        if not np.isfinite(wave).all():
            raise RuntimeError("Ming produced non-finite streaming audio")
        self.pending = np.concatenate((self.pending, wave))
        count = len(self.pending) // self.frame
        cut = 0
        if count:
            scores = np.abs(self.pending[:count * self.frame]).reshape(count, self.frame).mean(axis=1)
            voiced = np.flatnonzero(scores > 1e-3)
            if len(voiced):
                self.voiced = True
                cut = int(voiced[-1] + 1) * self.frame
        if final:
            if not self.voiced:
                raise RuntimeError("Ming produced silent audio while streaming")
            cut = min(count * self.frame, cut + self.tail)
        out, self.pending = self.pending[:cut], self.pending[cut:]
        return out


def _decode(ming, params, ctx):
    import torch

    model = ming.model
    pending = []
    cached = None
    state = (None, None, None)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for latent, last in model.sample(**params):
            if ctx is not None:
                ctx.checkpoint()
            pending.append(latent)
            if len(pending) > STREAM_FRAMES:
                batch = torch.cat(pending[:STREAM_FRAMES], dim=1)
                wave, state, cached = model.audio.decode(
                    batch, past_key_values=cached, use_cache=True,
                    stream_state=state, last_chunk=False)
                pending = pending[STREAM_FRAMES:]
                if wave.numel():
                    yield wave.detach().float().cpu().numpy().reshape(-1)
            if last:
                break
        if ctx is not None:
            ctx.checkpoint()
        if not pending:
            raise RuntimeError("Ming produced no acoustic latents")
        wave, _, _ = model.audio.decode(
            torch.cat(pending, dim=1), past_key_values=cached, use_cache=True,
            stream_state=state, last_chunk=True)
        if wave.numel():
            yield wave.detach().float().cpu().numpy().reshape(-1)


def iter_synthesize(ming, kwargs, ctx=None):
    import torch
    import tts_tasks

    params = dict(kwargs)
    steps = int(params.pop("max_decode_steps", 1000))
    request = tts_tasks.build_task_generation_kwargs(**params)
    with torch.inference_mode():
        prompt_latent, spk_emb = None, None
        if request["use_zero_spk_emb"]:
            spk_emb = [torch.zeros(1, 192)]
        elif request["prompt_wav_path"] is not None:
            prompt_latent, spk_emb = ming.prepare_prompt(
                request["prompt_wav_path"], request["use_spk_emb"])
        prompt_text = request["prompt_text"] if prompt_latent is not None else None
        if prompt_text is None:
            prompt_latent = None
        fragments = ming.prepare_text(
            request["text"], request["taskname"], request["text_language"], 24)
        zero_spk = request["use_zero_spk_emb"]
        for index, fragment in enumerate(fragments):
            if ctx is not None:
                ctx.checkpoint()
            args = dict(
                prompt=request["prompt"], text=fragment,
                spk_emb=None if spk_emb is None else [e.clone() for e in spk_emb],
                instruction=request["instruction"], prompt_latent=prompt_latent,
                prompt_text=prompt_text, max_decode_steps=steps, cfg=request["cfg"],
                sigma=request["sigma"], temperature=request["temperature"],
                use_zero_spk_emb=zero_spk)
            tail = SilenceTail(ming.sample_rate)
            reference = []
            for wave in _decode(ming, args, ctx):
                out = tail.push(wave)
                if len(out):
                    if request["taskname"] == "STYLE" and index == 0:
                        reference.append(out)
                    yield out
            out = tail.push([], final=True)
            if len(out):
                if request["taskname"] == "STYLE" and index == 0:
                    reference.append(out)
                yield out
            if request["taskname"] == "STYLE" and index == 0 and len(fragments) > 1:
                audio = torch.from_numpy(np.concatenate(reference)).reshape(1, -1)
                prompt_latent, spk_emb = ming._prepare_generated_prompt(audio)
                prompt_text, zero_spk = fragment, False
