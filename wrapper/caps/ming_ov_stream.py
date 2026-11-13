import os

import numpy as np

from .ming_stream import STREAM_FRAMES


class Upsample:
    def __init__(self, decoder):
        import torch.nn as nn
        import torch.nn.functional as F

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = decoder.fc1
                self.scale = max(1, int(decoder.patch_size))

            def forward(self, latent):
                hidden = self.fc(latent).transpose(1, 2)
                if self.scale > 1:
                    hidden = F.interpolate(hidden, scale_factor=self.scale, mode="linear", align_corners=False)
                return hidden[:, :, self.scale:-self.scale].transpose(1, 2)

        self.module = Module()


class Head:
    def __init__(self, head, mode):
        import torch.nn as nn

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = head

            def forward(self, hidden, *buffers):
                if mode == "whole":
                    return self.head(hidden, streaming=False)[0]
                audio = buffers[0] if buffers else None
                window = buffers[1] if buffers else None
                wave, _, audio, window = self.head(
                    hidden, streaming=True, audio_buffer=audio, window_buffer=window,
                    last_chunk=mode == "last")
                return wave, audio, window

        self.module = Module()


def _outputs(graph, *inputs):
    import torch
    from . import ming_ov

    compiled = graph.ensure()
    feed = {compiled.inputs[i]: ming_ov._numpy(t) for i, t in enumerate(inputs)}
    result = compiled(feed)
    return [torch.from_numpy(np.array(result[output], copy=True)) for output in compiled.outputs]


class Decoder:
    def __init__(self, upsample, runner, heads, scale):
        self.upsample, self.runner, self.heads = upsample, runner, heads
        self.scale = scale
        self.reset()

    def reset(self):
        self.runner.reset()
        self.pending = self.left = self.audio = self.window = None
        self.first = True

    def part(self, latent, right, final):
        import torch

        n = int(latent.shape[1])
        if not 0 < n <= STREAM_FRAMES:
            raise ValueError("invalid Ming streaming latent batch: %s" % n)
        left = latent[:, :1] if self.left is None else self.left
        padded = latent
        if n < STREAM_FRAMES:
            padded = torch.cat((latent, latent[:, -1:].expand(-1, STREAM_FRAMES - n, -1)), dim=1)
        hidden = _outputs(self.upsample, torch.cat((left, padded, right), dim=1))[0]
        hidden = self.runner.step(hidden[:, :n * self.scale])
        mode = ("whole" if final else "first") if self.first else ("last" if final else "middle")
        inputs = (hidden,) if self.first else (hidden, self.audio, self.window)
        output = _outputs(self.heads[mode, n], *inputs)
        if mode != "whole":
            self.audio, self.window = output[1:]
        self.left = latent[:, -1:]
        self.first = False
        return output[0]

    def __call__(self, latent, past_key_values=None, use_cache=True,
                 stream_state=(None, None, None), last_chunk=False):
        import torch

        if past_key_values is None:
            self.reset()
        waves = []
        if self.pending is not None:
            waves.append(self.part(self.pending, latent[:, :1], False))
        self.pending = latent
        if last_chunk:
            waves.append(self.part(latent, latent[:, -1:], True))
            self.reset()
        wave = torch.cat(waves, dim=-1) if waves else torch.empty(1, 1, 0)
        return wave, stream_state, ("ov-stream",)


def _runner(inner, src, device):
    import torch
    from .. import tts_ov

    layers, kv, dim = tts_ov.kv_meta(inner.config)
    hidden = int(inner.config.hidden_size)
    n = 16
    _, xml, stamp = tts_ov.ir_paths(src, "ming_vae_stream_prefill", ".ov-ming-vae-stream-prefill-v1")
    prefill = tts_ov.compile_causal(
        tts_ov.causal_kv_module(inner, False),
        (torch.zeros(1, n, hidden), torch.ones(1, n, dtype=torch.long)),
        xml, stamp, device, dynamize_ranks=(2, 3))
    past = tuple(torch.zeros(1, kv, n, dim) for _ in range(layers * 2))
    _, xml, stamp = tts_ov.ir_paths(src, "ming_vae_stream_decode", ".ov-ming-vae-stream-decode-v1")
    decode = tts_ov.compile_causal(
        tts_ov.causal_kv_module(inner, True, external_position=True),
        (torch.zeros(1, n, hidden), torch.ones(1, n * 2, dtype=torch.long),
         torch.arange(n, n * 2).reshape(1, n), *past),
        xml, stamp, device, dynamize_ranks=(2, 4))
    return tts_ov.DeviceKvRunner(decode, layers, prefill=prefill, external_position=True)


def install(bailing, src, device, props):
    import torch
    from . import ming_ov

    decoder = bailing.audio.decoder
    if decoder.semantic_model is not None:
        raise RuntimeError("Ming streaming VAE requires the acoustic decoder")
    scale = max(1, int(decoder.patch_size))
    hidden = int(decoder.decoder.config.hidden_size)
    latent = int(bailing.latent_dim)
    directory = os.path.join(src, "openvino")
    xml = os.path.join(directory, "vae_stream_upsample.xml")
    ming_ov._export(Upsample(decoder).module, torch.zeros(1, STREAM_FRAMES + 2, latent), xml)
    upsample = ming_ov._Graph(xml, device, props)
    upsample.ensure()
    heads = {}
    overlap = int(decoder.head.istft.buffer_len)
    for mode in ("first", "middle", "last", "whole"):
        counts = (STREAM_FRAMES,) if mode in ("first", "middle") else range(1, STREAM_FRAMES + 1)
        for n in counts:
            example = (torch.zeros(1, n * scale, hidden),)
            if mode in ("middle", "last"):
                example += (torch.zeros(1, overlap), torch.ones(1, overlap))
            xml = os.path.join(directory, "vae_stream_%s_%s.xml" % (mode, n))
            ming_ov._export(Head(decoder.head, mode).module, example, xml)
            heads[mode, n] = ming_ov._Graph(xml, device, props)
    heads["first", STREAM_FRAMES].ensure()
    heads["middle", STREAM_FRAMES].ensure()
    stream = Decoder(upsample, _runner(decoder.decoder, src, device), heads, scale)
    offline = bailing.audio.decode

    def decode(latent, past_key_values=None, use_cache=False,
               stream_state=(None, None, None), last_chunk=False):
        if use_cache:
            return stream(latent, past_key_values, use_cache, stream_state, last_chunk)
        return offline(latent, past_key_values, use_cache, stream_state, last_chunk)

    bailing.audio.decode = decode
    ming_ov.log.info("streaming VAE with device KV and overlap-add on %s", device)
