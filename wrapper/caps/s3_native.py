"""Product file and process boundaries around the unchanged official S3 runner."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import uuid


class OfficialEngine:
    sample_rate = 24000

    def __init__(self, snapshot, preset="adaptive"):
        self.source = Path("/opt/s3")
        self.root = Path(tempfile.gettempdir()) / "s3-official"
        self.preset = preset
        self.lock = threading.Lock()
        revision = (self.source / "COMMIT").read_text().strip()
        marker = self.root / ".source-revision"
        if not marker.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.source, self.root, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(".git", "cache", "models", "runs", "__pycache__"))
            marker.write_text(revision + "\n")
        if marker.read_text().strip() != revision:
            raise RuntimeError("The application directory contains a different official S3 source revision")
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
            installed.write_text(release_id)
            shutil.rmtree(staged)
        check = subprocess.run(
            ["audio-python", "-c", "from vllm.engine.omnidrca2_engine import OmniDRCA2Engine; "
             "import vllm; assert '/vendor/vllm/' in vllm.__file__; print(OmniDRCA2Engine.__module__)"],
            cwd=self.root, env=self.env, capture_output=True, text=True, timeout=120,
        )
        if check.returncode:
            raise RuntimeError("Official S3 engine import failed:\n" + check.stderr[-6000:])

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
            subprocess.run(
                ["audio-python", str(self.root / "tools/make_input.py"), "--audio", str(incoming),
                 "--direction", "zh2en" if source == "Chinese" else "en2zh", "--output", str(input_json)],
                cwd=self.root, env=self.env, check=True, capture_output=True,
            )
            output = job / "output"
            prepare = ["audio-python", str(self.root / "tools/infer.py"), "--mode", "streaming",
                       "--preset", self.preset, "--model", str(self.root / "models/streaming"),
                       "--input", str(input_json), "--output", str(output), "--gpu", "0",
                       "--no-warmup", "--dry-run"]
            with (job / "inference.log").open("w") as log:
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
                )
                for name, folder in {"TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor",
                                     "VLLM_CACHE_ROOT": "vllm", "CUDA_CACHE_PATH": "cuda", "TMPDIR": "tmp",
                                     "TORCH_EXTENSIONS_DIR": "torch_extensions"}.items():
                    config[name] = str(output / "cache" / folder)
                config.update(TORCHINDUCTOR_COMPILE_THREADS="1", MAX_JOBS="2")
                proc = subprocess.run(command, cwd=self.root, env=config, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode:
                raise RuntimeError("Official S3 inference failed; log: " + str(job / "inference.log"))
            rows = [json.loads(line) for line in (output / "raw.jsonl").read_text().splitlines() if line.strip()]
            if len(rows) != 1 or rows[0].get("error"):
                raise RuntimeError("Official S3 returned an incomplete or failed row: " + str(rows))
            row = rows[0]
            live = row.get("live_pcm") or {}
            wav = Path(live.get("wav") or "")
            if not wav.is_file() or wav.suffix != ".wav":
                raise RuntimeError("Official S3 did not produce its final WAV")
            return row, wav.read_bytes()
