# The audio.cpp server as a child engine: its spec, its config, its process, and the wire to it.
#
# Capability-agnostic on purpose. audio.cpp covers 47 model specs across TTS, ASR, alignment,
# separation, music and effects; this module knows how to run any of them and nothing about what
# they mean. Everything task-shaped lives in a cap module, so adding STT here is a new caps/ file
# plus two catalog entries, not a rewrite.
#
# What the model can do is read off the spec audio.cpp itself ships in the image rather than
# configured by us: `tasks` says whether cloning exists, `modes` says whether it can stream. A cap
# mounts routes from those facts, so the endpoint list can neither overpromise nor sell the model
# short, and a model added later is described by upstream instead of by our guesses.
import atexit
import glob
import json
import logging
import os
import signal
import subprocess
import threading
import time

import httpx

log = logging.getLogger("audio-acpp")

SERVER_BIN = "/app/audiocpp_server"
CLI_BIN = "/app/audiocpp_cli"
SPEC_DIR = "/app/model_specs"

# The id we register with audio.cpp. Callers never see it: every route answers as MODEL_NAME, and
# one instance holds one model, so a stable internal id keeps the config free of naming rules.
MODEL_ID = "engine"

# Enough for the process to bind and register its models; the model load is waited on separately.
BOOT_TIMEOUT_S = 120.0
# Synthesis of a long script on a busy card. The engine's own busy timeout is what queues requests.
REQUEST_TIMEOUT_S = 1800.0


class SpecError(RuntimeError):
    """The image has no spec for these weights, or more than one claims them."""


def _norm(value):
    return str(value or "").strip().lower()


def loader_advertisements(binary=CLI_BIN, timeout_s=60.0):
    """What every registered loader says it can do, asked of the engine itself.

    `audiocpp_cli --list-loaders --json` answers out of the loader code that will run the model,
    so it names the modes per task and carries instructions_policy — which the HTTP face never
    exposes, yet decides whether `instructions` does anything at all. No weights are touched.

    Returns {} rather than raising: the packaged spec still describes the model, and refusing to
    boot over a missing side-channel would be worse than losing the finer detail.
    """
    try:
        proc = subprocess.run([binary, "--list-loaders", "--json"], capture_output=True,
                              text=True, timeout=timeout_s)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("cannot ask %s for loader capabilities: %s", binary, e)
        return {}
    if proc.returncode != 0:
        log.warning("%s --list-loaders exited %s: %s", binary, proc.returncode,
                    (proc.stderr or "").strip()[:200])
        return {}
    try:
        doc = json.loads(proc.stdout)
    except ValueError as e:
        log.warning("cannot parse loader capabilities: %s", e)
        return {}
    return {_norm(f): row for f, row in (doc.get("loaders") or {}).items()}


class Spec:
    """What one audio.cpp model can do: its packaged spec, refined by the loader's own claims."""

    def __init__(self, family, doc, advert=None):
        self.family = family
        self.doc = doc
        self.advert = advert or {}
        self.category = str(doc.get("category") or "")
        self.languages = tuple(doc.get("languages") or ())
        self.display_name = str(doc.get("display_name") or family)
        # Packaged metadata: two flat lists, every task assumed to support every mode.
        self._spec_tasks = tuple(_norm(t) for t in (doc.get("tasks") or ()))
        self._spec_modes = tuple(_norm(m) for m in (doc.get("modes") or ()))
        # The loader's version: modes per task, which is what the server actually enforces.
        self.task_modes = {_norm(t): tuple(_norm(m) for m in (modes or ()))
                           for t, modes in (self.advert.get("tasks") or {}).items()}
        self.instructions_policy = _norm(self.advert.get("instructions_policy"))
        self.api_endpoints = tuple(self.advert.get("api_endpoints") or ())

    def __repr__(self):
        return "Spec(%s, tasks=%s, modes=%s, clone=%s)" % (
            self.family, list(self.tasks), list(self.modes), self.can_clone)

    @property
    def tasks(self):
        return tuple(self.task_modes) or self._spec_tasks

    @property
    def modes(self):
        """Every mode any task of this model supports."""
        if self.task_modes:
            return tuple(sorted({m for modes in self.task_modes.values() for m in modes}))
        return self._spec_modes

    @property
    def can_clone(self):
        """Zero-shot cloning, however this family happens to declare it.

        The two declarations are not interchangeable: some loaders advertise a voice_cloning task,
        others advertise plain TTS plus a speaker-reference flag that the CLI does not print. The
        packaged spec's clone entries cover the second kind.
        """
        if any("clon" in task for task in self.tasks):
            return True
        if any("clon" in task for task in self._spec_tasks):
            return True
        return bool((self.doc.get("capabilities") or {}).get("clone"))

    @property
    def can_design(self):
        """Voice from a natural-language description, with no reference audio."""
        return any("design" in task for task in self.tasks + self._spec_tasks)

    @property
    def streams(self):
        return "streaming" in self.modes

    @property
    def prefixes_instructions(self):
        """Whether `instructions` has to be folded into the text instead of sent as a field.

        VoxCPM2 is the case that matters: its session rejects style conditions outright and its
        policy is text_prefix, so an `instructions` field would be silently dropped — the caller
        would ask for a voice and get the default one with a 200.
        """
        return self.instructions_policy == "text_prefix"

    @property
    def run_mode(self):
        """The mode to configure, preferring streaming because it subsumes offline.

        A streaming-configured model still answers a plain POST /v1/audio/speech: the server
        collects its own stream and returns one buffer (handle_speech, app/server/runtime.cpp).
        Configuring offline instead would refuse every stream request, so one instance in
        streaming mode serves both shapes without a second copy of the weights.
        """
        return "streaming" if self.streams else "offline"

    @property
    def task(self):
        """What audio.cpp should run this model as. Cloning is not a task of its own: it is TTS
        with reference audio, which is why `clone` never reaches server.json."""
        for task in self.tasks:
            if "clon" not in task and "design" not in task:
                return task
        return self.category


def load_specs(spec_dir=SPEC_DIR, adverts=None):
    if adverts is None:
        adverts = loader_advertisements()
    out = {}
    for path in sorted(glob.glob(os.path.join(spec_dir, "*.json"))):
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError) as e:
            log.warning("skipping unreadable spec %s: %s", path, e)
            continue
        family = str(doc.get("family") or os.path.splitext(os.path.basename(path))[0])
        out[family] = Spec(family, doc, adverts.get(_norm(family)))
    return out


def _spec_filenames(spec):
    """Every weight filename this spec's packages name, basename only."""
    names = set()
    for pkg in spec.doc.get("packages") or ():
        for rel in pkg.get("files") or ():
            names.add(os.path.basename(rel))
    return names


def _claimants(directory, specs):
    """Which specs name a file that is actually in this directory."""
    try:
        on_disk = set(os.listdir(directory))
    except OSError:
        return []
    return sorted(s.family for s in specs.values() if _spec_filenames(s) & on_disk)


def _snapshot_root(repo, token=None):
    """Where the weights landed. llm-init's own answer wins when it left one."""
    # Written by llm-init after the download, and it already accounts for MODEL_SOURCE --subdir.
    try:
        with open("/run/llm-init/model_path", "r", encoding="utf-8") as f:
            path = f.read().strip()
        if path and os.path.isdir(path):
            return path
    except OSError:
        pass
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, local_files_only=True,
                             cache_dir=os.environ.get("HF_HUB_CACHE"), token=token)


def resolve_weights(repo, family="", token=None, spec_dir=SPEC_DIR):
    """(weights_dir, spec) for the pre-downloaded weights: llm-init fetches before we start.

    The spec is matched by filename rather than guessed from the model name: the name is ours and
    shows up in /v1/models, while the weight filenames come from the same spec that has to load
    them. One level below the snapshot root is searched too, because a GGUF repo holding many
    models keeps each in its own folder and MODEL_SOURCE --subdir is optional.
    """
    root = _snapshot_root(repo, token)
    specs = load_specs(spec_dir)
    if family and family not in specs:
        raise SpecError("no audio.cpp spec for family %r; this image ships %d specs (%s, ...)"
                        % (family, len(specs), ", ".join(sorted(specs)[:6])))
    candidates = [root] + sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p))
    for directory in candidates:
        hits = _claimants(directory, specs)
        if family:
            if family in hits:
                return directory, specs[family]
            continue
        if len(hits) == 1:
            return directory, specs[hits[0]]
        if len(hits) > 1:
            raise SpecError("%d specs claim the weights in %s (%s); name one with ENGINE_ARGS "
                            "--family" % (len(hits), directory, ", ".join(hits)))
    listing = ", ".join(sorted(os.listdir(root))[:8]) if os.path.isdir(root) else "unreadable"
    if family:
        raise SpecError("none of %s or its subdirectories holds a file that the %s spec names "
                        "(found: %s)" % (root, family, listing))
    raise SpecError("no audio.cpp spec claims any file under %s (found: %s); name the family in "
                    "ENGINE_ARGS, e.g. --family voxcpm2" % (root, listing))


class Engine:
    """The audio.cpp server process, its generated config, and typed access to its HTTP face."""

    def __init__(self, *, spec, weights_dir, args, work_dir="/tmp/acpp"):
        # args is the cap's EngineArgs: flags we need are claimed here, the rest reach the binary.
        self.spec = spec
        self.weights_dir = weights_dir
        self._args = args
        self._work_dir = work_dir
        self.port = args.count("--port", int(os.environ.get("ENGINE_PORT", "8000")) + 1)
        self.backend = str(args.text("--backend", _default_backend()))
        self.device = args.count("--device", 0)
        self.threads = args.count("--threads", _default_threads())
        # The engine queues concurrent work; this is how long a caller waits for its turn.
        self.busy_timeout_ms = args.count("--busy-timeout-ms", 600000)
        self._proc = None
        self._client = None
        self._config_path = None
        self._log_tail = []

    @property
    def base_url(self):
        return "http://127.0.0.1:%d" % self.port

    @property
    def alive(self):
        return self._proc is not None and self._proc.poll() is None

    def log_tail(self, limit=12):
        """The child's last lines, so a load failure reports its own reason and not just a code."""
        return "\n".join(self._log_tail[-limit:])

    def config_doc(self):
        """server.json. Only the engine's own vocabulary appears here, so upstream docs apply.

        No voice_presets and no voice_dir: with neither, GET /v1/audio/voices is empty, and a
        model without built-in voices should not advertise a picker. `voice` is never forwarded
        either (see the cap), because an unmatched name is read as a cached voice id, which some
        models reject outright.
        """
        return {
            "host": "127.0.0.1",
            "port": self.port,
            "backend": self.backend,
            "device": self.device,
            "threads": self.threads,
            # Load at startup, not on first request: readiness has to mean the weights are in.
            "lazy_load": False,
            "busy_timeout_ms": self.busy_timeout_ms,
            # The wrapper is the only public face; the bundled WebUI and its installer stay off.
            "ui_enabled": False,
            "ui_management": False,
            "models": [
                {
                    "id": MODEL_ID,
                    "family": self.spec.family,
                    "path": self.weights_dir,
                    "task": self.spec.task,
                    "mode": self.spec.run_mode,
                }
            ],
        }

    def argv(self):
        # Whatever no cap claimed goes to the binary in its own spelling: ENGINE_ARGS is the one
        # channel for engine-native flags, so a knob we never thought about still reaches it.
        return [SERVER_BIN, "--config", self._config_path, "--no-ui"] + list(
            self._args.passthrough())

    def start(self, timeout_s=BOOT_TIMEOUT_S):
        """Write the config, spawn the server, and return once it answers /health."""
        os.makedirs(self._work_dir, exist_ok=True)
        self._config_path = os.path.join(self._work_dir, "server.json")
        doc = self.config_doc()
        with open(self._config_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        log.info("audio.cpp config: family=%s task=%s mode=%s backend=%s device=%d threads=%d "
                 "path=%s", self.spec.family, self.spec.task, self.spec.run_mode, self.backend,
                 self.device, self.threads, self.weights_dir)
        argv = self.argv()
        log.info("starting %s", " ".join(argv))
        self._proc = subprocess.Popen(argv, cwd="/app", stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=self._pump_logs, daemon=True).start()
        atexit.register(self.stop)
        self._client = httpx.Client(base_url=self.base_url, timeout=REQUEST_TIMEOUT_S)
        self._await_health(timeout_s)

    def _pump_logs(self):
        """The child's stdout is the only account of a load failure, so it goes to ours."""
        for line in self._proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            self._log_tail.append(line)
            del self._log_tail[:-64]
            log.info("[audio.cpp] %s", line)

    def _await_health(self, timeout_s):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if not self.alive:
                raise RuntimeError("audio.cpp exited with %s during startup:\n%s"
                                   % (self._proc.returncode, self.log_tail()))
            try:
                r = self._client.get("/health", timeout=5.0)
                if r.status_code == 200:
                    log.info("audio.cpp is up on %s: %s", self.base_url, r.text.strip()[:200])
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise RuntimeError("audio.cpp did not answer /health within %.0fs:\n%s"
                           % (timeout_s, self.log_tail()))

    def stop(self):
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()

    def post(self, path, payload, timeout=REQUEST_TIMEOUT_S):
        """One request to the child. Raises EngineError with the engine's own message on failure."""
        if not self.alive:
            raise EngineError("audio.cpp is not running:\n%s" % self.log_tail(), status=503)
        try:
            r = self._client.post(path, json=payload, timeout=timeout)
        except httpx.HTTPError as e:
            raise EngineError("audio.cpp did not answer: %s" % e, status=502)
        if r.status_code >= 400:
            raise EngineError(_engine_message(r), status=r.status_code)
        return r

    def stream(self, path, payload, timeout=REQUEST_TIMEOUT_S):
        """A streaming POST, yielding raw response chunks. Caller parses the framing it asked for."""
        if not self.alive:
            raise EngineError("audio.cpp is not running:\n%s" % self.log_tail(), status=503)
        return self._client.stream("POST", path, json=payload, timeout=timeout)


class EngineError(RuntimeError):
    """A failure the child engine reported, with the status it used."""

    def __init__(self, message, status=500):
        super().__init__(message)
        self.status = status


def _engine_message(response):
    """audio.cpp answers errors as JSON when it can; fall back to the body it did send."""
    try:
        doc = response.json()
    except ValueError:
        return (response.text or "").strip()[:500] or "engine error %d" % response.status_code
    for key in ("error", "message", "detail"):
        val = doc.get(key) if isinstance(doc, dict) else None
        if isinstance(val, dict):
            val = val.get("message") or val.get("detail")
        if isinstance(val, str) and val.strip():
            return val.strip()
    return json.dumps(doc)[:500]


def _default_backend():
    """cuda when this container actually has a slice of a card, cpu when it was given none."""
    from . import gpu

    return "cuda" if gpu.quota_mib() or gpu.visible_memory_bytes() else "cpu"


def _default_threads():
    # ggml's CPU paths (audio i/o, the codec on cpu backends) scale with this; the engine's own
    # default of 1 leaves a multi-core node idle. Capped because more threads than cores thrashes.
    try:
        return max(1, min(8, len(os.sched_getaffinity(0))))
    except AttributeError:
        return max(1, min(8, os.cpu_count() or 1))
