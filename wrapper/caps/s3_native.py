"""Product file and process boundaries around the unchanged official S3 runner."""
import atexit
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import wave
import uuid


def _stage_source(source, root):
    revision = (source / "COMMIT").read_text().strip()
    marker = root / ".source-revision"
    layout = root / ".source-layout"
    if marker.exists() and marker.read_text().strip() != revision:
        raise RuntimeError("The application directory contains a different official S3 source revision")
    patches = source / "PATCHES.json"
    identity = hashlib.sha256(patches.read_bytes()).hexdigest() if patches.exists() else "native"
    layout_id = revision + ":3:" + identity
    if not layout.exists() or layout.read_text().strip() != layout_id:
        root.mkdir(parents=True, exist_ok=True)

        def ignored(directory, names):
            excluded = {".git", "__pycache__"}
            if Path(directory) == source:
                excluded.update(("cache", "models", "runs"))
            return set(names).intersection(excluded)

        shutil.copytree(source, root, dirs_exist_ok=True, ignore=ignored)
        marker.write_text(revision + "\n")
        layout.write_text(layout_id + "\n")


class OfficialEngine:
    sample_rate = 24000

    def __init__(self, snapshot, preset="adaptive", kv_cache_gib=0.5):
        self.source = Path("/opt/s3")
        self.root = Path(tempfile.gettempdir()) / "s3-official"
        self.preset = preset
        self.kv_cache_gib = float(kv_cache_gib)
        self.lock = threading.Lock()
        _stage_source(self.source, self.root)
        self.env = self._environment()
        self.env["STREAMING_FEATURE_EXTRACTOR"] = str(self.root / "models/streaming")
        installed = self.root / ".model-installed"
        release = Path(snapshot) / "streaming" / "release.json"
        if not release.is_file():
            raise RuntimeError("S3 streaming release.json is missing from the shared snapshot")
        release_id = release.read_text()
        if not installed.exists() or installed.read_text() != release_id:
            staged = self.root / "cache/model_downloads/streaming"
            shutil.copytree(Path(snapshot) / "streaming", staged, dirs_exist_ok=True)
            subprocess.run(
                ["audio-python", str(self.root / "tools/download_model.py"), "--mode", "streaming",
                 "--local-dir", str(staged)],
                cwd=self.root, env=self.env, check=True,
            )
            shutil.copyfile(release, self.root / "models/streaming/release.json")
            installed.write_text(release_id)
            shutil.rmtree(staged)
        model_release = self.root / "models/streaming/release.json"
        if not model_release.is_file():
            shutil.copyfile(release, model_release)
        self._start()

    def _start(self):
        service = self.root / "services" / uuid.uuid4().hex
        service.mkdir(parents=True)
        boot = service / "bootstrap.wav"
        with wave.open(str(boot), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\x00" * 3200)
        input_json = service / "input.jsonl"
        self._make_input(boot, input_json, "English")
        output = service / "output"
        prepare = ["audio-python", str(self.root / "tools/infer.py"), "--mode", "streaming",
                   "--preset", self.preset, "--model", str(self.root / "models/streaming"),
                   "--input", str(input_json), "--output", str(output), "--gpu", "0",
                   "--no-warmup", "--dry-run", "--kv-cache-gib", str(self.kv_cache_gib)]
        self.log_path = service / "inference.log"
        with self.log_path.open("w") as log:
            subprocess.run(prepare, cwd=self.root, env=self.env, check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        command = json.loads((output / "COMMAND.json").read_text())["command"]
        config = self.env.copy()
        config.update(
            OMNIDRCA2_MODEL_DIR=str(self.root / "models/streaming"),
            TUNE_ROOT=str(output), TUNE_CONFIG=str(output / "sampling.json"),
            SINGLE_GPU_PHYSICAL_ID="0", SINGLE_CAPTURE_AUDIT=str(output / "hidden_audit.jsonl"),
            S3_POLICY_CONFIG=str(output / "policy.json"), CODEX_TALKER_MAX_CODEC_TEXT_RATIO="0",
            CODEX_AUDIO_BLANK_TOKEN_COUNT="5", CODEX_ALIGNMENT_MASK_TOKEN_ID="-60",
            FLEURS_KV_CACHE_BYTES=str(int(self.kv_cache_gib * 1024**3)),
        )
        for name, folder in {"TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor",
                             "VLLM_CACHE_ROOT": "vllm", "CUDA_CACHE_PATH": "cuda", "TMPDIR": "tmp",
                             "TORCH_EXTENSIONS_DIR": "torch_extensions"}.items():
            config[name] = str(output / "cache" / folder)
        config.update(TORCHINDUCTOR_COMPILE_THREADS="1", MAX_JOBS="2",
                      S3_NATIVE_OVERRIDE_DIR=str(self.root / "inference/streaming/vllm_overrides"))
        self.socket_path = str(service / "engine.sock")
        ready = service / "ready.json"
        worker_config = service / "worker.json"
        worker_config.write_text(json.dumps({"command": command, "socket": self.socket_path,
                                             "ready": str(ready)}))
        config["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + ":" + config.get("PYTHONPATH", "")
        with self.log_path.open("a") as log:
            self.process = subprocess.Popen(
                ["audio-python", "-m", "wrapper.caps.s3_native_worker", str(worker_config)], cwd=self.root, env=config,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        atexit.register(self.close)
        deadline = time.monotonic() + 3300
        while not ready.exists():
            if self.process.poll() is not None:
                raise RuntimeError("Official S3 startup failed:\n" + self.log_path.read_text()[-6000:])
            if time.monotonic() > deadline:
                self.close()
                raise TimeoutError("Official S3 startup timed out; log: " + str(self.log_path))
            time.sleep(0.5)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def _make_input(self, incoming, output, source):
        subprocess.run(
            ["audio-python", str(self.root / "tools/make_input.py"), "--audio", str(incoming),
             "--direction", "zh2en" if source == "Chinese" else "en2zh", "--output", str(output)],
            cwd=self.root, env=self.env, check=True, capture_output=True,
        )

    def _environment(self):
        proc = subprocess.run(
            ["bash", "-c", '. "$1/env.sh"; env -0', "s3-env", str(self.root)],
            capture_output=True, check=True,
        )
        return dict(record.decode().split("=", 1) for record in proc.stdout.split(b"\0") if b"=" in record)

    def translate(self, audio, source, target):
        if (source, target) not in (("Chinese", "English"), ("English", "Chinese")):
            raise ValueError("S3 supports Chinese to English and English to Chinese")
        with self.lock:
            job = self.root / "requests" / uuid.uuid4().hex
            job.mkdir(parents=True)
            incoming = job / (job.name + ".wav")
            incoming.write_bytes(audio)
            input_json = job / "input.jsonl"
            self._make_input(incoming, input_json, source)
            row = json.loads(input_json.read_text().strip())
            if self.process.poll() is not None:
                raise RuntimeError("Official S3 worker stopped; log: " + str(self.log_path))
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(3300)
                connection.connect(self.socket_path)
                with connection.makefile("rwb") as channel:
                    channel.write((json.dumps({"row": row}) + "\n").encode())
                    channel.flush()
                    response = json.loads(channel.readline())
            if response.get("error"):
                raise RuntimeError("Official S3 inference failed: " + response["error"])
            result = response["result"]
            live = result.get("live_pcm") or {}
            wav = Path(live.get("wav") or "")
            if not wav.is_file() or wav.suffix != ".wav":
                raise RuntimeError("Official S3 did not produce its final WAV")
            (job / "result.json").write_text(json.dumps(result, ensure_ascii=False))
            return result, wav.read_bytes()
