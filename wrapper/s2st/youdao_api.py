"""Pinned author's CPU speech bridge with audited upstream delivery adapters."""

import argparse
from pathlib import Path
import sys
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--translation-chunk-ms", type=int, default=0, choices=(0, 2000, 4000))
    parser.add_argument("--overlap-translation", action="store_true")
    parser.add_argument("--share-upstream-gpu", action="store_true")
    parser.add_argument("--share-asr-ming-gpu", action="store_true")
    options = parser.parse_args()
    if options.share_upstream_gpu and options.share_asr_ming_gpu:
        raise ValueError("Only one GPU stage pair can share admission")
    if (options.share_upstream_gpu or options.share_asr_ming_gpu) and options.overlap_translation:
        raise ValueError("Shared upstream GPU cannot bypass MING admission")
    from .gpu_queue import configure

    configure(options.root / "gpu-stage-fifo", shared_upstream=options.share_upstream_gpu,
              shared_asr_ming=options.share_asr_ming_gpu)
    sys.path.insert(0, "/opt/youdao/t3po")
    import httpx

    original_post = httpx.Client.post

    def reject_legacy_completion(self, url, *args, **kwargs):
        target = urlparse(str(url))
        if target.hostname in ["127.0.0.1", "localhost"] and target.port == 18092:
            raise RuntimeError(
                "Only the observed incremental translation path is allowed; retries are disabled"
            )
        return original_post(self, url, *args, **kwargs)

    httpx.Client.post = reject_legacy_completion
    from .youdao_bounded_wait import install as bounded
    from .youdao_source import install as source
    from .youdao_target import install as target

    bounded(8.0)
    source(options.translation_chunk_ms)
    target(options.root / "results/target-stream-observer.jsonl", options.overlap_translation)
    from inference.server import main as official_main

    official_main()


if __name__ == "__main__":
    main()
