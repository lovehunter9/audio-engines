"""Keep the author's complete streaming runner resident behind a serial socket."""
import importlib
import json
from pathlib import Path
import signal
import socket
import sys
import time
import traceback


def serve(config_path):
    config = json.loads(Path(config_path).read_text())
    command = config["command"]
    sys.path.insert(0, str(Path(command[1]).parent))
    sys.argv = command[1:] + ["--end", "0"]
    import numpy as np
    entry = importlib.import_module(Path(command[1]).stem)
    entry = getattr(entry, "frozen", entry)
    module = entry.common.m
    import os
    os.environ["S3_NATIVE_TRANSPORT"] = "local"
    from .s3_inproc import install
    install()
    from .s3_native_compat import install_worker_compat
    install_worker_compat()
    from vllm.engine.arg_utils import AsyncEngineArgs
    original_post_init = AsyncEngineArgs.__post_init__

    def post_init(self):
        original_post_init(self)
        self.worker_cls = "wrapper.caps.s3_cuda_worker.NativeWorker"
        self.distributed_executor_backend = "uni"

    AsyncEngineArgs.__post_init__ = post_init
    backend_type = module.StreamingS2STBackend
    encoder_type = module.s2tt.FullRecordEncoder
    resident = {}

    class Encoder(encoder_type):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            resident["encoder"] = self

    class Backend(backend_type):
        def __init__(self, args):
            super().__init__(args)
            from .s3_inproc import use_original_eager_pcm
            use_original_eager_pcm(self.pcm)
            resident.update(backend=self, args=args)

        def close(self):
            pass

    module.s2tt.FullRecordEncoder = Encoder
    module.StreamingS2STBackend = Backend
    try:
        module.main()
    finally:
        module.s2tt.FullRecordEncoder = encoder_type
        module.StreamingS2STBackend = backend_type
    backend, encoder, args = (resident[name] for name in ("backend", "encoder", "args"))
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(config["socket"])
    Path(config["socket"]).chmod(0o600)
    server.listen(1)
    Path(config["ready"]).write_text(json.dumps({"ready": True}))

    def stop(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    index = 0
    try:
        while True:
            connection, _ = server.accept()
            with connection, connection.makefile("rwb") as channel:
                request = json.loads(channel.readline())
                try:
                    started = time.monotonic()
                    backend.start_sample()
                    result = module.s2tt.evaluate_one_streaming_thinker_deferred_talker(
                        index, request["row"], encoder, backend, args,
                    )
                    codec, segments = backend.finish_sample()
                    codec_path = Path(args.codec_dir) / (result["key"] + ".npy")
                    np.save(codec_path, codec)
                    np.save(codec_path.with_suffix(".control.npy"), backend._global_codec_full)
                    result.update(codec_path=str(codec_path), codec_len=int(len(codec)),
                                  codec_segments=segments, elapsed_s=time.monotonic() - started)
                    response = {"result": result}
                except Exception as error:
                    traceback.print_exc()
                    response = {"error": str(error) or repr(error)}
                index += 1
                channel.write((json.dumps(response, ensure_ascii=False) + "\n").encode())
                channel.flush()
    finally:
        server.close()
        backend_type.close(backend)


if __name__ == "__main__":
    try:
        serve(sys.argv[1])
    except Exception:
        traceback.print_exc()
        raise
