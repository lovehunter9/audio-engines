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
import socket
import subprocess
import threading
import time
from urllib.parse import urlencode

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


def materialize_weights(directory, spec, work_dir="/tmp/acpp"):
    """A directory whose weight files are real files, which is what audio.cpp needs to load them.

    The engine resolves the path it is handed with weakly_canonical() and then picks the loader
    purely from the extension (open_tensor_source, src/framework/assets/tensor_source.cpp). A
    HuggingFace snapshot is a tree of symlinks into a blob store where each blob is named by its
    hash and has no extension at all, so following the link turns `model-bf16.gguf` into
    `blobs/db68...-10` and the engine refuses it as an "unsupported tensor source format" — with
    the weights sitting right there, fully downloaded.

    Hard links fix that for free: they carry the name, share the bytes, and canonicalize to
    themselves. They only work within one filesystem, so a copy into work_dir is the fallback for
    the case where the cache and the scratch space are different mounts.
    """
    wanted = _spec_filenames(spec)
    try:
        names = sorted(set(os.listdir(directory)) & wanted)
    except OSError:
        return directory
    if not names or not any(os.path.islink(os.path.join(directory, n)) for n in names):
        return directory
    real = {n: os.path.realpath(os.path.join(directory, n)) for n in names}
    # Beside the blob store rather than in it: same filesystem (so linking works), and outside the
    # models--*/ trees huggingface_hub prunes.
    cache_root = os.environ.get("HF_HUB_CACHE") or os.path.dirname(os.path.dirname(
        os.path.dirname(next(iter(real.values())))))
    for target in (os.path.join(cache_root, ".audiocpp-weights", os.path.basename(directory)),
                   os.path.join(work_dir, "weights", os.path.basename(directory))):
        try:
            os.makedirs(target, exist_ok=True)
            for name in names:
                dst, src = os.path.join(target, name), real[name]
                if os.path.exists(dst) and os.stat(dst).st_ino == os.stat(src).st_ino:
                    continue  # already linked to this exact blob
                if os.path.lexists(dst):
                    os.unlink(dst)
                os.link(src, dst)
            log.info("linked %d weight file(s) into %s so the real path keeps its extension",
                     len(names), target)
            return target
        except OSError as e:
            log.info("cannot hard-link weights into %s (%s); trying a copy", target, e)
    target = os.path.join(work_dir, "weights", os.path.basename(directory))
    os.makedirs(target, exist_ok=True)
    import shutil

    for name in names:
        dst, src = os.path.join(target, name), real[name]
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            continue
        shutil.copy2(src, dst)
    log.warning("copied %d weight file(s) into %s: hard links were not possible, so this used "
                "disk equal to the weights", len(names), target)
    return target


def _source_flags():
    """`--include` / `--subdir` from MODEL_SOURCE: llm-init already honoured these when it fetched.

    The shared HuggingFace cache for `audio-cpp/audio.cpp-gguf` holds every family anyone on this
    node has downloaded, each in its own folder. Without these flags the first folder that matches
    a spec wins, which is how a VoxCPM2 install ended up loading MOSS-TTS-Nano.
    """
    tokens = os.environ.get("MODEL_SOURCE", "").split()
    includes, subdir = [], ""
    i = 0
    while i < len(tokens):
        if tokens[i] == "--include" and i + 1 < len(tokens):
            includes.append(tokens[i + 1])
            i += 2
            continue
        if tokens[i] == "--subdir" and i + 1 < len(tokens):
            subdir = tokens[i + 1]
            i += 2
            continue
        i += 1
    return includes, subdir


def _snapshot_root(repo, token=None):
    """Where the weights landed. llm-init's own answer wins when it left one."""
    # Preferred when present because it already accounts for MODEL_SOURCE's --include/--subdir.
    # llm-init sometimes writes the file itself (a single GGUF), not the folder, so a file is
    # accepted and we load from its directory.
    try:
        with open("/run/llm-init/model_path", "r", encoding="utf-8") as f:
            path = f.read().strip()
        if path and os.path.isfile(path):
            return os.path.dirname(path)
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
    includes, subdir = _source_flags()
    include_names = {os.path.basename(p) for p in includes}

    candidates = [root]
    if subdir:
        nested = os.path.join(root, subdir)
        if os.path.isdir(nested):
            candidates.append(nested)
    candidates.extend(sorted(p for p in glob.glob(os.path.join(root, "*"))
                             if os.path.isdir(p) and p not in candidates))
    # A --include file is the one llm-init actually fetched; keep only folders that hold it.
    if include_names:
        holding = []
        for directory in candidates:
            try:
                names = set(os.listdir(directory))
            except OSError:
                continue
            if include_names & names:
                holding.append(directory)
        if holding:
            candidates = holding

    singles = []
    for directory in candidates:
        hits = _claimants(directory, specs)
        if family:
            if family in hits:
                return directory, specs[family]
            continue
        if len(hits) == 1:
            singles.append((directory, hits[0]))
        elif len(hits) > 1:
            raise SpecError("%d specs claim the weights in %s (%s); name one with ENGINE_ARGS "
                            "--family" % (len(hits), directory, ", ".join(hits)))
    if len(singles) == 1:
        return singles[0][0], specs[singles[0][1]]
    if len(singles) > 1:
        families = ", ".join(f for _d, f in singles)
        raise SpecError("%d model folders under %s (%s); this GGUF repo holds many families. "
                        "Name one with ENGINE_ARGS --family, or --include the file in MODEL_SOURCE"
                        % (len(singles), root, families))
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
            # Dictation / a voice-agent turn can outlast the engine's 10-minute live default.
            "live_ingest": {"total_timeout_ms": 1800000},
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
        # Done here rather than in resolve_weights: reading what a model can do has to stay free of
        # side effects, since the routes are built from it before anything is started.
        self.weights_dir = materialize_weights(self.weights_dir, self.spec, self._work_dir)
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

    def stream_live(self, path, content, params=None, timeout=REQUEST_TIMEOUT_S):
        """Chunked PCM in, SSE out: the engine's /v1/audio/transcriptions/live shape.

        httpx's HTTP/1.1 client finishes the request body before it reads the
        response, so every partial arrives after `stop`. This opens one TCP
        socket, writes chunks on a side thread, and reads SSE on the caller
        thread — the engine is localhost, so one thread send / one thread recv
        is enough. A buffered (not chunked) body is a 400 from the engine.
        """
        if not self.alive:
            raise EngineError("audio.cpp is not running:\n%s" % self.log_tail(), status=503)
        return LiveDuplex("127.0.0.1", self.port, path, content, params or {}, timeout)


class LiveDuplex:
    """One HTTP/1.1 request that is still sending while the response is read."""

    def __init__(self, host, port, path, content, params, timeout):
        self.status_code = 0
        self._host = host
        self._port = int(port)
        self._path = path if path.startswith("/") else "/" + path
        self._content = content
        self._params = params or {}
        self._timeout = timeout
        self._sock = None
        self._writer = None
        self._leftover = b""
        self._chunked = False

    def __enter__(self):
        qs = urlencode(self._params) if self._params else ""
        target = "%s?%s" % (self._path, qs) if qs else self._path
        req = (
            "POST %s HTTP/1.1\r\n"
            "Host: %s:%d\r\n"
            "Content-Type: application/octet-stream\r\n"
            "Transfer-Encoding: chunked\r\n"
            "Accept: text/event-stream\r\n"
            "Connection: close\r\n"
            "\r\n"
        ) % (target, self._host, self._port)
        self._sock = socket.create_connection((self._host, self._port), timeout=self._timeout)
        self._sock.settimeout(self._timeout)
        self._sock.sendall(req.encode("ascii"))
        self._writer = threading.Thread(target=self._write_body, daemon=True)
        self._writer.start()
        self.status_code, self._leftover, self._chunked = self._read_headers()
        return self

    def __exit__(self, *exc):
        try:
            if self._sock is not None:
                self._sock.close()
        except OSError:
            pass
        self._sock = None
        if self._writer is not None:
            self._writer.join(timeout=2.0)
        return False

    def iter_bytes(self):
        buf = self._leftover
        self._leftover = b""
        if self._chunked:
            yield from self._iter_chunked(buf)
            return
        if buf:
            yield buf
        while True:
            chunk = self._recv()
            if not chunk:
                return
            yield chunk

    def read(self):
        return b"".join(self.iter_bytes())

    def _write_body(self):
        try:
            for piece in self._content:
                if piece:
                    self._send_chunk(piece)
            self._send_chunk(b"")
        except OSError:
            return

    def _send_chunk(self, data):
        if self._sock is None:
            raise OSError("live socket closed")
        if data:
            self._sock.sendall(("%x\r\n" % len(data)).encode("ascii") + data + b"\r\n")
        else:
            self._sock.sendall(b"0\r\n\r\n")

    def _recv(self):
        if self._sock is None:
            return b""
        try:
            return self._sock.recv(65536)
        except (OSError, socket.timeout):
            return b""

    def _read_headers(self):
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self._recv()
            if not chunk:
                raise EngineError("audio.cpp closed before live response headers", status=502)
            buf += chunk
        raw, rest = buf.split(b"\r\n\r\n", 1)
        lines = raw.split(b"\r\n")
        first = lines[0].decode("ascii", "replace")
        parts = first.split()
        if len(parts) < 2:
            raise EngineError("audio.cpp sent a bad live status line: %s" % first, status=502)
        try:
            status = int(parts[1])
        except ValueError:
            raise EngineError("audio.cpp sent a bad live status line: %s" % first, status=502)
        headers = {}
        for line in lines[1:]:
            if b":" not in line:
                continue
            key, val = line.split(b":", 1)
            headers[key.decode("ascii", "replace").lower()] = val.strip().decode("ascii", "replace")
        chunked = "chunked" in headers.get("transfer-encoding", "").lower()
        return status, rest, chunked

    def _iter_chunked(self, buf):
        while True:
            while b"\r\n" not in buf:
                more = self._recv()
                if not more:
                    return
                buf += more
            line, buf = buf.split(b"\r\n", 1)
            try:
                size = int(line.split(b";", 1)[0], 16)
            except ValueError:
                return
            if size == 0:
                return
            while len(buf) < size + 2:
                more = self._recv()
                if not more:
                    if buf:
                        yield buf
                    return
                buf += more
            piece, buf = buf[:size], buf[size:]
            if buf.startswith(b"\r\n"):
                buf = buf[2:]
            if piece:
                yield piece


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
