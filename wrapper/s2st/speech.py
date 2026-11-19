import dataclasses
import hashlib
import json


def settings(url):
    from simulinfer.config import Settings

    return Settings(
        tts_backend="vllm_omni",
        tts_url=url,
        tts_model="Ming-3.0-tts-0.5B",
        tts_task_type="Base",
        tts_clone_source=True,
        tts_history_mode="none",
        tts_history_window_s=10,
        tts_history_max_chars=240,
        tts_min_chars=40,
        tts_batch_ms=650,
        tts_clone_first_seconds=2.4,
        tts_clone_interval_seconds=10,
        tts_tail_filter=False,
        tts_speed=1.1,
        tts_speed_zh_en=1.2,
        tts_timeout_seconds=60,
    )


def fingerprint(value):
    values = {k: v for k, v in dataclasses.asdict(value).items() if k.startswith("tts_")}
    # Loopback addresses are deployment details, not synthesis behavior.
    values.pop("tts_url", None)
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def install():
    from .delivery import install as delivery
    from .clause import install as clause

    delivery()
    clause()
    import fireredvad
    import torch

    if getattr(fireredvad.FireRedStreamVad, "_s2st_cuda", False):
        return
    original = fireredvad.FireRedStreamVad.from_pretrained.__func__

    def create(cls, model_dir, config=None):
        config = config or fireredvad.FireRedStreamVadConfig()
        config.use_gpu = True
        model = original(cls, model_dir, config)
        devices = {
            str(p.device)
            for item in vars(model).values()
            if isinstance(item, torch.nn.Module)
            for p in item.parameters()
        }
        if not devices or any(not value.startswith("cuda") for value in devices):
            raise RuntimeError("Source VAD neural weights are not wholly CUDA")
        return model

    fireredvad.FireRedStreamVad.from_pretrained = classmethod(create)
    fireredvad.FireRedStreamVad._s2st_cuda = True
