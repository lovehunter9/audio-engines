from contextlib import contextmanager


class _CachedCodec:
    def __init__(self, decoder, stream_chunk_size=None):
        import torch

        self.torch = torch
        self.decoder = decoder
        self.codes = None
        self.audio = None
        self.quant_tail = None
        self.hidden_tail = None
        self.cache = None
        self.position = 0
        self.stream_chunk_size = stream_chunk_size
        self.calls = 0

    def _advance(self, codes, context):
        torch, decoder = self.torch, self.decoder
        count = codes.shape[-1]
        quant = decoder.quantizer.decode(codes)
        if self.quant_tail is not None:
            quant = torch.cat([self.quant_tail, quant], dim=-1)
        padding = int(decoder.pre_conv.padding)
        self.quant_tail = quant[..., -padding:].clone() if padding else None
        hidden = decoder.pre_conv(quant)[..., -count:].transpose(1, 2)
        transformed = decoder.pre_transformer(
            inputs_embeds=hidden, past_key_values=self.cache, use_cache=True,
            cache_position=torch.arange(self.position, self.position + count, device=hidden.device))
        self.cache = transformed.past_key_values
        self.position += count
        hidden = transformed.last_hidden_state.transpose(1, 2)
        if self.hidden_tail is not None:
            hidden = torch.cat([self.hidden_tail, hidden], dim=-1)
        self.hidden_tail = hidden[..., -context:].clone() if context else None
        for blocks in decoder.upsample:
            for block in blocks:
                hidden = block(hidden)
        for block in decoder.decoder:
            hidden = block(hidden)
        return hidden[..., -count * int(decoder.total_upsample):].clamp(-1, 1)

    def decode(self, codes, chunk_size=300, left_context_size=25):
        count = codes.shape[-1]
        seen = self.codes.shape[-1] if self.codes is not None else 0
        window = (self.stream_chunk_size is not None and
                  self.calls * self.stream_chunk_size >= max(self.stream_chunk_size, int(left_context_size)))
        if not seen:
            overlap = 0
        elif not window and count >= seen and self.torch.equal(codes[..., :seen], self.codes):
            overlap = seen
        else:
            overlap = min(int(left_context_size), seen, count)
            if not overlap or not self.torch.equal(codes[..., :overlap], self.codes[..., -overlap:]):
                raise ValueError("codec window does not overlap the previous decoding")
        start = seen - overlap
        new_codes = codes[..., overlap:]
        for index in range(0, new_codes.shape[-1], int(chunk_size)):
            audio = self._advance(new_codes[..., index:index + int(chunk_size)], int(left_context_size))
            self.audio = audio if self.audio is None else self.torch.cat([self.audio, audio], dim=-1)
        self.codes = codes if self.codes is None else self.torch.cat([self.codes, new_codes], dim=-1)
        self.calls += 1
        factor = int(self.decoder.total_upsample)
        return self.audio[..., start * factor:(start + count) * factor]


@contextmanager
def causal_decode(model, chunk_size=None):
    tokenizer = getattr(model, "speech_tokenizer", None)
    decoder = getattr(getattr(tokenizer, "model", None), "decoder", None)
    if decoder is None or not hasattr(decoder, "pre_transformer"):
        yield
        return
    had_override = "chunked_decode" in vars(decoder)
    original = decoder.chunked_decode
    decoder.chunked_decode = _CachedCodec(decoder, chunk_size).decode
    try:
        yield
    finally:
        if had_override:
            decoder.chunked_decode = original
        else:
            del decoder.chunked_decode
