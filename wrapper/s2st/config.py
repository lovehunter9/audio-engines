from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Models:
    translation: Path
    speech: Path
    recognition: Path | None = None

    def validate(self, family):
        if family not in ("tencent", "youdao"):
            raise ValueError("unknown cascade")
        if self.translation.suffix.lower() != ".gguf" or not self.translation.is_file():
            raise ValueError("translation must be a local GGUF file")
        if not self.speech.is_dir():
            raise ValueError("speech must be a local MING snapshot")
        if family == "youdao" and (self.recognition is None or not self.recognition.is_dir()):
            raise ValueError("Youdao requires a local R2T2 snapshot")


@dataclass(frozen=True)
class Translation:
    context: int
    batch: int | None = None
    microbatch: int | None = None
    threads: int = 12
    kv_type: str | None = None

    def __post_init__(self):
        if min(self.context, self.threads) <= 0:
            raise ValueError("translation limits must be positive")
        if (self.batch is None) != (self.microbatch is None):
            raise ValueError("batch and microbatch must be configured together")
        if self.batch is not None and (
            min(self.batch, self.microbatch) <= 0 or self.microbatch > self.batch
        ):
            raise ValueError("invalid batch sizes")
        if self.kv_type not in (None, "q8_0", "f16"):
            raise ValueError("unvalidated KV type")

    def arguments(self, path):
        result = (
            "-m",
            str(path),
            "-c",
            str(self.context),
            "-ngl",
            "999",
            "-np",
            "1",
            "-t",
            str(self.threads),
            "-fa",
            "on",
            "--override-tensor",
            "token_embd.weight=CUDA0",
        )
        if self.batch is not None:
            result += ("-b", str(self.batch), "-ub", str(self.microbatch))
        if self.kv_type is not None:
            result += ("-ctk", self.kv_type, "-ctv", self.kv_type)
        return result


TENCENT_TRANSLATION = Translation(context=10240, batch=128, microbatch=64, kv_type="q8_0")
YOUDAO_TRANSLATION = Translation(context=8192)

TENCENT_MODEL_REF = "8d4e9381419ac17d17e29359a297109afce02066"
MING_MODEL_REF = "8d66b70a00f05ad183fa8a863b4ce609fc83cbc0"

YOUDAO_MODEL_REF = "86d1098305f8acf2f025d6c4359640fb657ec491"
R2T2_MODEL_REF = "d07b4a593610c16074cfc941fba052a7a12eeda3"
VAD_MODEL_REF = "7990aaccc6b7aec1e527743bd30201f2c4a03b8c"
