"""One OpenVINO checkpoint in this process, then exit so the GPU pool is returned."""
import json
import struct
import sys
from pathlib import Path

import numpy as np

from .qwen3_voice import _apply_seed, _load_faster, _pair, _stream_item, bake_native


def _write_frame(fh, audio, sr):
    raw = np.asarray(audio, dtype="<f4").reshape(-1).tobytes()
    fh.write(struct.pack("<II", int(sr), len(raw) // 4))
    fh.write(raw)
    fh.flush()


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if job["method"] == "bake":
        bake_native(job["dest"])
        return
    model = _load_faster(job["repo"])
    kwargs = dict(job["kwargs"])
    _apply_seed(kwargs.pop("seed", None))
    method = job["method"]
    if job.get("stream") or method.endswith("_streaming"):
        # Library warnings print on stdout. Frames go to a file so they stay aligned.
        with open(job["frames"], "r+b", buffering=0) as fh:
            for item in getattr(model, method)(**kwargs):
                audio, sr = _stream_item(item)
                if len(audio):
                    _write_frame(fh, audio, sr)
            fh.write(struct.pack("<II", 0, 0))
            fh.flush()
        return
    audio, sr = _pair(getattr(model, method)(**kwargs))
    out = Path(job["out"])
    np.save(out, np.asarray(audio, dtype="float32"))
    out.with_suffix(".sr").write_text(str(int(sr)), encoding="utf-8")


if __name__ == "__main__":
    main()
