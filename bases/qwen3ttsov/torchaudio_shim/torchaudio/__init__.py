# Pure-Python stand-in. The CPU index has no torchaudio build for torch 2.14,
# and the 2.9 wheel will not load against it. The 12 Hz Qwen3-TTS path never
# calls into this package; the 25 Hz tokenizer imports it at startup.
__version__ = "0"
