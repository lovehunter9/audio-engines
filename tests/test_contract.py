import glob
import importlib
import json
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from wrapper import catalog, contract, gpu, tasks

EXPECTED_CAPABILITY_ENDPOINTS = {
    ("align", "align", "POST", "/v1/audio/align"): {"async_supported": True},
    # No WS: ggml synthesizes whole utterances, so stream=1 is sentence-scoped.
    ("crispasr_tts", "tts", "GET", "/v1/audio/voices"): {"async_supported": False},
    ("crispasr_tts", "tts", "POST", "/v1/audio/speech"): {"async_supported": True},
    ("crispasr_tts", "tts", "POST", "/v1/audio/speech/batch"): {
        "async_supported": True,
    },
    ("diar", "diar", "POST", "/v1/audio/diarization"): {"async_supported": True},
    # Same capability, same route, different engine: speakrs serves diar from its own image.
    ("diar_speakrs", "diar", "POST", "/v1/audio/diarization"): {"async_supported": True},
    ("diar_stream", "diar_stream", "WS", "/v1/audio/diarize/stream"): {
        "async_supported": False,
    },
    ("embed", "speaker_embed", "POST", "/v1/audio/embeddings"): {
        "async_supported": True,
    },
    ("enhance", "enhance", "POST", "/v1/audio/enhance"): {"async_supported": True},
    ("sound_fx", "sound_fx", "POST", "/v1/audio/speech"): {"async_supported": True},
    ("sound_fx", "sound_fx", "POST", "/v1/audio/speech/batch"): {
        "async_supported": True,
    },
    ("stt_stream", "stt", "POST", "/v1/audio/transcriptions"): {
        "async_supported": True,
    },
    ("stt_stream", "stt_stream", "WS", "/v1/audio/stream"): {
        "async_supported": False,
    },
    ("tts", "tts", "GET", "/v1/audio/voices"): {"async_supported": False},
    ("tts", "tts", "POST", "/v1/audio/speech"): {"async_supported": True},
    ("tts", "tts", "POST", "/v1/audio/speech/batch"): {"async_supported": True},
    ("tts", "tts", "WS", "/v1/audio/speech/stream"): {"async_supported": False},
    ("tts", "tts_clone", "POST", "/v1/audio/speech"): {"async_supported": True},
    ("tts", "tts_clone", "POST", "/v1/audio/speech/batch"): {
        "async_supported": True,
    },
    ("tts", "tts_clone", "POST", "/v1/audio/speech/clone"): {
        "async_supported": True,
    },
    ("tts", "tts_clone", "WS", "/v1/audio/speech/stream"): {
        "async_supported": False,
    },
    ("tts_dialogue", "tts_dialogue", "POST", "/v1/audio/speech"): {
        "async_supported": True,
    },
    ("vad", "vad", "POST", "/v1/audio/vad"): {"async_supported": True},
    ("whisper", "stt", "POST", "/v1/audio/transcriptions"): {
        "async_supported": True,
    },
    ("whisper", "stt", "POST", "/v1/audio/translations"): {
        "async_supported": True,
    },
}

_EL_TTS_ROUTES = (
    ("GET", "/v1/voices", False),
    ("GET", "/v1/voices/settings/default", False),
    ("GET", "/v1/voices/{voice_id}", False),
    ("GET", "/v1/voices/{voice_id}/settings", False),
    ("POST", "/v1/voices/{voice_id}/settings/edit", False),
    ("DELETE", "/v1/voices/{voice_id}", False),
    ("POST", "/v1/voices/{voice_id}/edit", False),
    ("POST", "/v1/text-to-speech/{voice_id}", True),
    ("POST", "/v1/text-to-speech/{voice_id}/stream", False),
    ("GET", "/v1/history", False),
    ("POST", "/v1/history/download", False),
    ("GET", "/v1/history/{history_item_id}", False),
    ("GET", "/v1/history/{history_item_id}/audio", False),
    ("DELETE", "/v1/history/{history_item_id}", False),
)
_EL_DESIGN_ROUTES = (
    ("POST", "/v1/text-to-voice/design", True),
    ("POST", "/v1/text-to-voice", True),
)
_EL_CLONE_ROUTES = (
    ("POST", "/v1/voices/add", True),
)
for _el_mod in ("firered", "breeze"):
    for _method, _path, _async in _EL_TTS_ROUTES:
        EXPECTED_CAPABILITY_ENDPOINTS[(_el_mod, "tts", _method, _path)] = {
            "async_supported": _async,
        }
    for _method, _path, _async in _EL_DESIGN_ROUTES:
        EXPECTED_CAPABILITY_ENDPOINTS[(_el_mod, "tts_design", _method, _path)] = {
            "async_supported": _async,
        }
    for _method, _path, _async in _EL_CLONE_ROUTES:
        EXPECTED_CAPABILITY_ENDPOINTS[(_el_mod, "tts_clone", _method, _path)] = {
            "async_supported": _async,
        }

TASK_PATHS = {
    "/v1/tasks",
    "/v1/tasks/{id}",
    "/v1/tasks/{id}/result",
}


class CatalogContractTest(unittest.TestCase):
    def test_shared_paths_keep_model_specific_operation_semantics(self):
        self.assertEqual(
            catalog.describe_endpoint("tts_dialogue", "tts_dialogue", "POST", "/v1/audio/speech", True)["operation_id"],
            "speech.dialogue",
        )
        self.assertEqual(
            catalog.describe_endpoint("sound_fx", "sound_fx", "POST", "/v1/audio/speech", True)["operation_id"],
            "sound.generate",
        )
        clone = catalog.describe_endpoint("tts", "tts_clone", "POST", "/v1/audio/speech", True)
        self.assertEqual(clone["operation_id"], "speech.synthesize.reference")
        self.assertNotEqual(clone["operation_id"], "voice.design.preview")
        persisted = catalog.describe_endpoint("tts", "tts_clone", "POST", "/v1/audio/speech/clone", True)
        self.assertEqual(persisted["operation_id"], "speech.synthesize.reference")
        self.assertEqual(persisted["resource_scope"], "model")
        for module in ("firered", "breeze"):
            synth = catalog.describe_endpoint(module, "tts", "POST", "/v1/text-to-speech/{voice_id}", True)
            design = catalog.describe_endpoint(module, "tts_design", "POST", "/v1/text-to-voice/design", True)
            self.assertEqual(synth["protocol"], "elevenlabs.voice.v1")
            self.assertEqual(synth["operation_id"], "speech.synthesize")
            self.assertEqual(design["operation_id"], "voice.design.preview")

    def test_register_does_not_mutate_shared_contract_endpoints(self):
        original = [dict(endpoint) for endpoint in contract.CONTRACT_ENDPOINTS]

        for model_name in ("first-model", "second-model"):
            app = FastAPI(title=model_name)
            contract.register(
                app,
                model_name=model_name,
                module="stt_stream",
                served=["stt"],
                is_ready=lambda: True,
                task_api=True,
            )

        self.assertEqual(contract.CONTRACT_ENDPOINTS, original)

    def test_engine_spec_can_report_dynamic_operation_availability(self):
        app = FastAPI()
        with mock.patch.dict(os.environ, {"AUDIO_BASE": "qwen3tts"}):
            contract.register(
                app, model_name="demo", module="tts", served=["tts"], is_ready=lambda: True,
                endpoint_available=lambda endpoint: (
                    (False, "checkpoint has no preset voice library")
                    if endpoint.get("operation_id") == "voice.list" else (True, "")
                ),
            )
        spec = TestClient(app).get("/api/engine-spec").json()
        voice_list = next(row for row in spec["endpoints"] if row.get("operation_id") == "voice.list")
        self.assertFalse(voice_list["available"])
        self.assertEqual(voice_list["reason"], "checkpoint has no preset voice library")

    def test_catalog_imports_without_engine_dependencies(self):
        modules = {
            module
            for entries in catalog.BASES.values()
            for _caps, module in entries
        }

        imported = {
            importlib.import_module(f"wrapper.caps.{module}").__name__
            for module in modules
        }

        self.assertEqual(imported, {f"wrapper.caps.{module}" for module in modules})

    def test_every_catalog_capability_has_its_complete_endpoint_mapping(self):
        catalog_pairs = {
            (module, cap)
            for entries in catalog.BASES.values()
            for caps, module in entries
            for cap in caps
        }
        expected_pairs = {
            (module, cap)
            for module, cap, _method, _path in EXPECTED_CAPABILITY_ENDPOINTS
        }
        self.assertEqual(catalog_pairs, expected_pairs)
        self.assertEqual(catalog_pairs, set(catalog._MOUNTS))
        self.assertEqual(
            set(catalog.FAMILIES),
            {module for entries in catalog.BASES.values() for _caps, module in entries},
        )

        expected_mounts = {}
        for module, cap, method, path in EXPECTED_CAPABILITY_ENDPOINTS:
            expected_mounts.setdefault((module, cap), set()).add((method, path))
        actual = {
            pair: {(method, path) for method, path, _description, _async in mounts}
            for pair, mounts in catalog._MOUNTS.items()
        }
        self.assertEqual(actual, expected_mounts)

        actual_metadata = {}
        for module, cap in catalog_pairs:
            for endpoint in catalog.endpoints(module, [cap]):
                key = (module, cap, endpoint["method"], endpoint["path"])
                metadata = {"async_supported": endpoint["async_supported"]}
                actual_metadata[key] = metadata
        self.assertEqual(actual_metadata, EXPECTED_CAPABILITY_ENDPOINTS)

    def test_el_instance_does_not_advertise_the_same_route_twice(self):
        for module in ("firered", "breeze"):
            rows = catalog.endpoints(module, ["tts", "tts_clone", "tts_design"])
            keys = [(row["method"], row["path"]) for row in rows]
            self.assertEqual(keys, list(dict.fromkeys(keys)), module)

    def test_task_advertisements_use_the_llm_init_contract_literals(self):
        advertised = {endpoint["path"] for endpoint in tasks.ENDPOINTS if not endpoint.get("deprecated")}
        self.assertEqual(advertised, TASK_PATHS)
        self.assertEqual(tasks.TASKS_PATH, "/v1/tasks")
        self.assertIn("/v1/tasks/{id}", advertised)
        self.assertIn("/v1/tasks/{id}/result", advertised)

    def test_supports_prefix_round_trips_through_the_public_model_capabilities(self):
        capabilities = list(dict.fromkeys(
            cap
            for entries in catalog.BASES.values()
            for caps, _module in entries
            for cap in caps
        ))
        keys = ["supports_" + cap for cap in capabilities]
        with mock.patch.dict(os.environ, {"MODEL_SUPPORTS": ",".join(keys)}, clear=False):
            bare, invalid = contract.parse_supports()
            payload = contract.models_payload("test-model", [contract.COARSE_CAPABILITY] + bare)

        self.assertEqual(invalid, [])
        self.assertEqual(bare, capabilities)
        self.assertEqual(["supports_" + cap for cap in bare], keys)
        self.assertEqual(payload["models"][0]["capabilities"], ["audio"] + capabilities)


class RuntimeHelperTest(unittest.TestCase):
    def test_runtime_parses_model_source_port_and_mutable_state(self):
        from wrapper.runtime import Runtime

        with mock.patch.dict(
            os.environ,
            {
                "MODEL_NAME": "served-name",
                "MODEL_SOURCE": "hf://org/repo",
                "ENGINE_PORT": "9123",
                "LOG_LEVEL": "WARNING",
            },
            clear=False,
        ):
            engine = Runtime(model=None)

        self.assertEqual(engine.model_name, "served-name")
        self.assertEqual(engine.model_repo, "org/repo")
        self.assertEqual(engine.port, 9123)
        self.assertEqual(engine.log_level, "warning")
        self.assertEqual(engine.state, {"ready": False, "error": None, "model": None})
        engine.state.update(ready=True)
        self.assertTrue(engine.state["ready"])

    def test_runtime_does_not_invent_a_name_or_repo(self):
        from wrapper.runtime import Runtime

        with mock.patch.dict(
            os.environ,
            {"MODEL_NAME": "", "MODEL_SOURCE": ""},
            clear=False,
        ):
            engine = Runtime()
        self.assertEqual(engine.model_name, "")
        self.assertEqual(engine.model_repo, "")

    def test_runtime_does_not_treat_model_name_as_a_repo(self):
        from wrapper.runtime import Runtime

        with mock.patch.dict(
            os.environ,
            {"MODEL_NAME": "chart-model", "MODEL_SOURCE": ""},
            clear=False,
        ):
            engine = Runtime()
        self.assertEqual(engine.model_name, "chart-model")
        self.assertEqual(engine.model_repo, "")

    def test_runtime_preserves_background_and_blocking_startup_axes(self):
        from wrapper.runtime import Runtime

        events = []
        background = Runtime()
        app = object()

        class FakeThread:
            def __init__(self, target, daemon):
                self.target = target
                self.daemon = daemon

            def start(self):
                events.append("thread.start")

        with (
            mock.patch("wrapper.runtime.threading.Thread", FakeThread),
            mock.patch(
                "wrapper.runtime.watchdog.arm",
                side_effect=lambda *args: events.append("watchdog"),
            ),
            mock.patch(
                "wrapper.runtime.uvicorn.run",
                side_effect=lambda *args, **kwargs: events.append(("uvicorn", kwargs)),
            ),
        ):
            def load():
                events.append("load")

            def build_app(supports):
                events.append(("build", supports))
                return app

            background.serve(["vad"], load, build_app, "background")

        self.assertEqual(
            events,
            [
                "thread.start",
                "watchdog",
                ("build", ["vad"]),
                (
                    "uvicorn",
                    {
                        "host": "0.0.0.0",
                        "port": background.port,
                        "log_level": background.log_level,
                    },
                ),
            ],
        )

        events.clear()
        blocking = Runtime()
        with (
            mock.patch(
                "wrapper.runtime.watchdog.arm",
                side_effect=lambda *args: events.append("watchdog"),
            ),
            mock.patch(
                "wrapper.runtime.uvicorn.run",
                side_effect=lambda *args, **kwargs: events.append(("uvicorn", kwargs)),
            ),
        ):
            blocking.serve(
                ["stt_stream"],
                lambda: events.append("load"),
                lambda supports: events.append(("build", supports)) or app,
                "blocking",
                load_on_main=True,
                disable_ws_ping=True,
            )

        self.assertEqual(
            events,
            [
                "watchdog",
                "load",
                ("build", ["stt_stream"]),
                (
                    "uvicorn",
                    {
                        "host": "0.0.0.0",
                        "port": blocking.port,
                        "log_level": blocking.log_level,
                        "ws_ping_interval": None,
                        "ws_ping_timeout": None,
                    },
                ),
            ],
        )

    def test_each_capability_keeps_its_runtime_wiring(self):
        cases = {
            "align": {
                "supports": ["align"],
                "watchdog": "Qwen3-ForcedAligner",
                "state": {"ready": False, "error": None, "model": None, "device": "cpu"},
            },
            "diar_speakrs": {
                "supports": ["diar"],
                "watchdog": "speakrs engine",
                "model": "speakrs-community-1",
                "repo": "speakrs-community-1",
                "state": {
                    "ready": False,
                    "error": None,
                    "pipeline": None,
                    "device": "cpu",
                    "params": {},
                },
            },
            "diar": {
                "supports": ["diar"],
                "watchdog": "pyannote pipeline",
                "state": {
                    "ready": False,
                    "error": None,
                    "pipeline": None,
                    "device": "cpu",
                    "batch1": False,
                    "params": {},
                },
            },
            "diar_stream": {
                "supports": ["diar_stream"],
                "watchdog": "streaming sortformer",
                "state": {"ready": False, "error": None, "model": None, "device": "cpu"},
                "disable_ws_ping": True,
            },
            "embed": {
                "supports": ["speaker_embed"],
                "watchdog": "pyannote embedding",
                "state": {
                    "ready": False,
                    "error": None,
                    "inference": None,
                    "device": "cpu",
                    "dim": None,
                },
            },
            "enhance": {
                "supports": ["enhance"],
                "watchdog": "speechbrain enhancement",
                "state": {
                    "ready": False,
                    "error": None,
                    "model": None,
                    "kind": None,
                    "device": "cpu",
                },
            },
            "stt_stream": {
                "supports": ["stt", "stt_stream"],
                "watchdog": "qwen-asr vLLM",
                "state": {"ready": False, "error": None, "asr": None},
                "load_on_main": True,
                "disable_ws_ping": True,
            },
            "vad": {
                "supports": ["vad"],
                "watchdog": "silero-vad",
                "state": {"ready": False, "error": None, "model": None, "get_ts": None},
            },
            "whisper": {
                "supports": ["stt"],
                "watchdog": "faster-whisper",
                "state": {
                    "ready": False,
                    "error": None,
                    "model": None,
                    "pipeline": None,
                    "device": None,
                    "compute": None,
                },
            },
        }

        for name, expected in cases.items():
            with self.subTest(capability=name):
                with mock.patch.dict(
                    os.environ,
                    {
                        "MODEL_NAME": "chart-model",
                        "MODEL_SOURCE": "hf://org/weights",
                        "ENGINE_PORT": "8000",
                        "LOG_LEVEL": "info",
                    },
                    clear=False,
                ):
                    module = importlib.reload(
                        importlib.import_module(f"wrapper.caps.{name}")
                    )
                    self.assertTrue(callable(module.build_app))
                    self.assertTrue(callable(module.run))
                    self.assertEqual(module.MODEL_NAME, "chart-model")
                    self.assertEqual(module._runtime.model_repo, "org/weights")
                    self.assertEqual(module._runtime.port, 8000)
                    self.assertEqual(module._runtime.log_level, "info")
                    self.assertEqual(module._state, expected["state"])
                    with mock.patch.object(module._runtime, "serve") as serve:
                        module.run(expected["supports"])

                args, kwargs = serve.call_args
                self.assertEqual(args[0], expected["supports"])
                self.assertTrue(callable(args[1]))
                self.assertTrue(callable(args[2]))
                self.assertEqual(args[3], expected["watchdog"])
                if name != "stt_stream":
                    self.assertIs(args[1], module._load)
                    self.assertIs(args[2], module.build_app)
                else:
                    with (
                        mock.patch.object(
                            module,
                            "_load_blocking",
                            side_effect=RuntimeError("load failed"),
                        ),
                        mock.patch.object(module, "_p"),
                        mock.patch.object(module.log, "exception"),
                    ):
                        args[1]()
                    self.assertEqual(module._state["error"], "load failed")
                    self.assertFalse(module._state["ready"])

                    app = object()
                    with (
                        mock.patch.object(module, "build_app", return_value=app) as build_app,
                        mock.patch.object(module, "_p"),
                    ):
                        self.assertIs(args[2](expected["supports"]), app)
                    build_app.assert_called_once_with(expected["supports"])
                self.assertEqual(
                    kwargs.get("load_on_main", False),
                    expected.get("load_on_main", False),
                )
                self.assertEqual(
                    kwargs.get("disable_ws_ping", False),
                    expected.get("disable_ws_ping", False),
                )


class SharedHelperTest(unittest.TestCase):
    def test_audioio_import_does_not_require_numpy(self):
        sys.modules.pop("wrapper.audioio", None)
        real_import = __import__

        def reject_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("numpy intentionally unavailable")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=reject_numpy):
            module = importlib.import_module("wrapper.audioio")

        self.assertTrue(callable(module.pcm16_to_float32))
        self.assertTrue(callable(module.resample_linear))

    def test_pcm16_conversion_handles_empty_and_little_endian_input(self):
        from wrapper.audioio import pcm16_to_float32

        empty = pcm16_to_float32(b"")
        self.assertEqual(empty.dtype.name, "float32")
        self.assertEqual(empty.shape, (0,))

        samples = pcm16_to_float32(b"\x00\x80\x00\x00\xff\x7f")
        self.assertEqual(samples.dtype.name, "float32")
        self.assertEqual(samples.tolist(), [-1.0, 0.0, 32767.0 / 32768.0])

    def test_same_rate_resampling_preserves_float32_and_copy_semantics(self):
        import numpy as np

        from wrapper.audioio import resample_linear

        samples = np.array([0.0, 0.5, 1.0], dtype="float32")
        same = resample_linear(samples, 16000)
        self.assertIs(same, samples)
        self.assertEqual(same.dtype.name, "float32")

        float64 = samples.astype("float64")
        converted = resample_linear(float64, 16000)
        self.assertIsNot(converted, float64)
        self.assertEqual(converted.dtype.name, "float32")
        self.assertEqual(converted.tolist(), samples.tolist())

    def test_linear_resampling_uses_python_round_for_target_length(self):
        import numpy as np

        from wrapper.audioio import resample_linear

        self.assertEqual(
            resample_linear(np.ones(1, dtype="float32"), 32000).shape,
            (0,),
        )
        self.assertEqual(
            resample_linear(np.ones(3, dtype="float32"), 32000).shape,
            (2,),
        )

    def test_different_rate_resampling_keeps_empty_arrays_float32(self):
        import numpy as np

        from wrapper.audioio import resample_linear

        empty = resample_linear(np.array([], dtype="float64"), 8000)
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(empty.dtype.name, "float32")

    def test_linear_resampling_interpolates_from_original_sample_points(self):
        import numpy as np

        from wrapper.audioio import resample_linear

        samples = np.array([0.0, 1.0], dtype="float32")
        upsampled = resample_linear(samples, 2, target_rate=4)
        self.assertEqual(upsampled.dtype.name, "float32")
        self.assertEqual(upsampled.tolist(), [0.0, 0.5, 1.0, 1.0])

    def test_segments_parser_keeps_existing_400_details(self):
        from wrapper.batch import parse_segments

        try:
            json.loads("{")
        except json.JSONDecodeError as e:
            expected_detail = f"invalid `segments` json: {e}"

        with self.assertRaises(HTTPException) as invalid:
            parse_segments("{")
        self.assertEqual(invalid.exception.status_code, 400)
        self.assertEqual(invalid.exception.detail, expected_detail)

        with self.assertRaises(HTTPException) as not_array:
            parse_segments('{"start": 0, "end": 1}')
        self.assertEqual(not_array.exception.status_code, 400)
        self.assertEqual(not_array.exception.detail, "`segments` must be a JSON array")
        self.assertEqual(parse_segments('[{"start": 0, "end": 1}]'), [{"start": 0, "end": 1}])


class _FakeUpload:
    """An UploadFile as far as the spill path is concerned: a name and chunked reads."""

    def __init__(self, total, filename="clip.wav", chunk=None):
        self.filename = filename
        self._left = total
        self._chunk = chunk or (1 << 20)
        self.reads = 0

    async def read(self, size=-1):
        self.reads += 1
        want = self._chunk if size is None or size < 0 else min(size, self._chunk)
        want = min(want, self._left)
        self._left -= want
        return b"\0" * want


class UploadBoundsTest(unittest.IsolatedAsyncioTestCase):
    """What one request may bring to the caps that hold a whole clip at once.

    Both bounds have to be enforced before the thing they are protecting against has happened.
    A byte count checked after `await file.read()` is a check on memory already spent, and a
    duration checked after the decode is one the OOM kill beat to it.
    """

    def _bounds(self, raw):
        from wrapper.contract import EngineArgs
        from wrapper.limits import Bounds

        return Bounds(EngineArgs(raw), seconds=14400)

    async def test_an_oversized_upload_stops_being_read_when_it_crosses_the_limit(self):
        from wrapper import limits

        bounds = self._bounds("--max-upload-mb 1")
        upload = _FakeUpload(total=64 * limits.MIB, chunk=1 << 20)
        with self.assertRaises(HTTPException) as refused:
            await bounds.spill(upload)
        self.assertEqual(refused.exception.status_code, 413)
        # Two reads: the one that fits and the one that crosses. The other 62 MiB never arrive,
        # which is the difference between this and a size check on an already-read body.
        self.assertEqual(upload.reads, 2)

    async def test_a_refused_upload_leaves_nothing_behind(self):
        from wrapper import limits

        bounds = self._bounds("--max-upload-mb 1")
        before = set(glob.glob(os.path.join(tempfile.gettempdir(), "upload-*")))
        with self.assertRaises(HTTPException):
            await bounds.spill(_FakeUpload(total=4 * limits.MIB))
        after = set(glob.glob(os.path.join(tempfile.gettempdir(), "upload-*")))
        self.assertEqual(after - before, set())

    async def test_a_short_file_that_decodes_into_hours_is_refused_on_its_duration(self):
        """The byte bound cannot catch this one: a compressed container is a fraction of its PCM."""
        bounds = self._bounds("--max-audio-seconds 60")
        upload = _FakeUpload(total=4096, filename="meeting.opus")
        with mock.patch("wrapper.limits.duration", return_value=9000.0):
            with self.assertRaises(HTTPException) as refused:
                await bounds.spill(upload)
        self.assertEqual(refused.exception.status_code, 413)
        self.assertIn("9000s", refused.exception.detail)
        self.assertIn("60s", refused.exception.detail)

    async def test_an_unmeasurable_container_is_passed_through_rather_than_refused(self):
        """A reader that cannot measure a file is not evidence about the engine that decodes it."""
        from wrapper.audioio import unlink

        bounds = self._bounds("--max-audio-seconds 60")
        with mock.patch("wrapper.limits.duration", return_value=None):
            path, seconds = await bounds.spill(_FakeUpload(total=2048))
        try:
            self.assertIsNone(seconds)
            self.assertTrue(os.path.exists(path))
        finally:
            unlink(path)

    async def test_the_bounds_are_engine_args_a_deployment_can_move(self):
        bounds = self._bounds("--max-audio-seconds 30 --max-upload-mb 7")
        self.assertEqual(bounds.seconds, 30.0)
        self.assertEqual(bounds.megabytes, 7.0)

    def test_every_cap_that_holds_a_whole_clip_declares_a_bound(self):
        """The point of the shared helper is that no cap here is the one that forgot.

        Read as source rather than imported: these modules import torch and pyannote at module
        scope, which the test image does not carry.
        """
        for module in ("diar", "diar_speakrs", "embed", "enhance", "vad"):
            root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            with open(os.path.join(root, "wrapper", "caps", module + ".py")) as f:
                src = f.read()
            self.assertIn("Bounds(_args", src, "%s declares no upload bound" % module)
            self.assertNotIn("await file.read()", src,
                             "%s reads the whole upload before bounding it" % module)


class TaskDurationTest(unittest.TestCase):
    """What a task document may say about duration, and when.

    A caller bills from these numbers, and several caps measure the input
    before they produce anything, so the status is what decides whether the
    measurement describes work that has a result.
    """

    def _task(self, status):
        from wrapper import tasks

        t = tasks.Task("stt", "m", lambda ctx: None)
        t.status = status
        tasks._Ctx(t).meter(input_seconds=12.5, output_seconds=3.0)
        return t

    def test_a_succeeded_task_reports_what_it_measured(self):
        doc = self._task("succeeded").doc()
        self.assertEqual(doc["input_duration_seconds"], 12.5)
        self.assertEqual(doc["output_duration_seconds"], 3.0)

    def test_a_task_that_did_not_succeed_reports_no_duration(self):
        for status in ("running", "failed", "canceled"):
            doc = self._task(status).doc()
            self.assertNotIn("input_duration_seconds", doc, status)
            self.assertNotIn("output_duration_seconds", doc, status)

    def test_metering_is_additive_and_ignores_nonsense(self):
        from wrapper import tasks

        t = tasks.Task("stt", "m", lambda ctx: None)
        ctx = tasks._Ctx(t)
        ctx.meter(input_seconds=1.5)
        ctx.meter(input_seconds=2.25)
        ctx.meter(input_seconds=-1)
        ctx.meter(input_seconds="not a number")
        ctx.meter(output_seconds=None)
        self.assertEqual(t.input_seconds, 3.75)
        self.assertIsNone(t.output_seconds)


class EngineSurfaceTest(unittest.TestCase):
    def setUp(self):
        self.ready = False
        self.env = mock.patch.dict(
            os.environ,
            {"AUDIO_BASE": "qwen", "MODEL_SUPPORTS": "supports_stt,supports_align"},
            clear=False,
        )
        self.env.start()
        self.addCleanup(self.env.stop)

        app = FastAPI(title="contract test")
        contract.register(
            app,
            model_name="test-model",
            module="stt_stream",
            served=["stt"],
            is_ready=lambda: self.ready,
            task_api=True,
        )
        gpu.mount_metrics(app)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_models_transitions_from_503_to_200(self):
        self.assertEqual(self.client.get("/v1/models").status_code, 503)

        self.ready = True
        response = self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["models"][0]["name"], "test-model")
        self.assertEqual(response.json()["data"][0]["id"], "test-model")

    def test_engine_spec_has_the_versioned_contract_shape(self):
        spec = self.client.get("/api/engine-spec").json()

        self.assertEqual(spec["schema_version"], 2)
        self.assertEqual(spec["base"], "qwen")
        self.assertEqual(spec["model"], "test-model")
        self.assertEqual(spec["implements"], ["stt", "stt_stream", "align"])
        self.assertEqual(spec["declares"], ["stt", "align"])
        self.assertEqual(spec["serves"], ["stt"])
        self.assertIsInstance(spec["endpoints"], list)
        self.assertTrue(spec["endpoints"])
        advertised = {endpoint["path"] for endpoint in spec["endpoints"]}
        self.assertTrue(TASK_PATHS.issubset(advertised))
        self.assertTrue(all("async_supported" in endpoint for endpoint in spec["endpoints"]))
        self.assertTrue(all("operation_id" in endpoint for endpoint in spec["endpoints"]))
        self.assertTrue(all("protocol" in endpoint for endpoint in spec["endpoints"]))
        by_route = {
            (endpoint["method"], endpoint["path"]): endpoint
            for endpoint in spec["endpoints"]
        }
        self.assertFalse(by_route[("GET", "/v1/models")]["async_supported"])
        for method, path in (
            ("GET", "/v1/tasks"),
            ("GET", "/v1/tasks/{id}"),
            ("GET", "/v1/tasks/{id}/result"),
            ("DELETE", "/v1/tasks/{id}"),
        ):
            self.assertFalse(by_route[(method, path)]["async_supported"])

        transcription = next(
            endpoint
            for endpoint in spec["endpoints"]
            if endpoint["path"] == "/v1/audio/transcriptions"
        )
        self.assertTrue(transcription["async_supported"])
        self.assertEqual(transcription["operation_id"], "audio.transcribe")
        self.assertEqual(transcription["protocol"], "openai.audio.v1")
        self.assertTrue(all("max_input_seconds" not in endpoint for endpoint in spec["endpoints"]))

    def test_metrics_expose_exactly_the_four_generic_gpu_gauges(self):
        response = self.client.get("/metrics")
        expected = {
            "gpu_present",
            "gpu_mem_used_bytes",
            "gpu_mem_total_bytes",
            "gpu_util_ratio",
        }
        samples = {
            line.split()[0]
            for line in response.text.splitlines()
            if line and not line.startswith("#")
        }
        type_declarations = {
            line
            for line in response.text.splitlines()
            if line.startswith("# TYPE ")
        }

        self.assertEqual(samples, expected)
        self.assertEqual(type_declarations, {f"# TYPE {name} gauge" for name in expected})


class RouteSpecAgreementTest(unittest.TestCase):
    """What the app mounts and what /api/engine-spec says it mounts.

    Nothing downstream can check this. llm-init relays the spec and the
    gateway routes on it, so a row that says available for a path FastAPI
    never mounted turns into a bare 404 two hops away, and a path that is
    mounted but absent from the spec is a capability the gateway will
    refuse as undeclared. Both look like an outage in somebody else's
    service.
    """

    # Mounted, and deliberately not advertised as available. Every entry is
    # a decision that has to be re-made when the route moves, which is why
    # this is a list of (module, capability, method, path) and not a
    # loosened assertion.
    MOUNTED_WITHOUT_ADVERTISING = frozenset({
        # FireRedTTS3-Instruct and Breeze TTS 2 serve preset, clone and
        # design off one weight, so the design routes are mounted whenever
        # tts is. tts_design is what declares them, and an instance that
        # declares only supports_tts leaves them mounted and unadvertised
        # (wrapper/catalog.py, the firered entry). The gateway routes on
        # what is advertised, so this is a route a caller cannot reach --
        # deliberately, and only reachable by declaring the capability.
        ("firered", "tts_design", "POST", "/v1/text-to-voice"),
        ("firered", "tts_design", "POST", "/v1/text-to-voice/design"),
        ("breeze", "tts_design", "POST", "/v1/text-to-voice"),
        ("breeze", "tts_design", "POST", "/v1/text-to-voice/design"),
        # Qwen3-TTS reads its preset library off the checkpoint, so
        # voice.list is mounted always and available only for a checkpoint
        # that has one. This is the endpoint_available callback working.
        ("tts", "tts", "GET", "/v1/audio/voices"),
    })

    @staticmethod
    def _placeholders(path):
        """`/v1/tasks/{tid}` and `/v1/tasks/{id}` are the same route.

        FastAPI names a path parameter after the handler argument; the
        catalog names it after the contract. Comparing the names would fail
        on a rename that changes nothing a caller can see.
        """
        return re.sub(r"\{[^}]*\}", "{}", path)

    @classmethod
    def _mounted(cls, app):
        from starlette.routing import WebSocketRoute

        out = set()
        for route in app.routes:
            path = getattr(route, "path", "")
            if not path.startswith("/v1/"):
                continue
            if isinstance(route, WebSocketRoute):
                out.add(("WS", cls._placeholders(path)))
                continue
            for method in getattr(route, "methods", None) or ():
                # Starlette adds these for free; the catalog does not
                # describe them and neither does the gateway route them.
                if method in ("HEAD", "OPTIONS"):
                    continue
                out.add((method, cls._placeholders(path)))
        return out

    def _spec(self, app):
        with TestClient(app) as client:
            endpoints = client.get("/api/engine-spec").json()["endpoints"]
        rows = [e for e in endpoints if e["path"].startswith("/v1/")]
        available = {
            (e["method"], self._placeholders(e["path"])) for e in rows if e["available"]
        }
        described = {(e["method"], self._placeholders(e["path"])) for e in rows}
        return available, described

    def _build(self, base, caps, module):
        env = {
            "AUDIO_BASE": base,
            "MODEL_SUPPORTS": ",".join(contract.SUPPORTS_PREFIX + c for c in caps),
        }
        with mock.patch.dict(os.environ, env, clear=False):
            built = importlib.import_module("wrapper.caps." + module)
            return built.build_app(list(caps))

    def test_every_base_mounts_exactly_what_it_advertises(self):
        for base, routes in catalog.BASES.items():
            for caps, module in routes:
                with self.subTest(base=base, module=module):
                    app = self._build(base, caps, module)
                    mounted = self._mounted(app)
                    available, described = self._spec(app)

                    self.assertEqual(
                        available - mounted,
                        set(),
                        f"{base} advertises routes it does not mount",
                    )
                    self.assertEqual(
                        mounted - described,
                        set(),
                        f"{base} mounts routes its spec does not describe",
                    )
                    allowed = {
                        (method, self._placeholders(path))
                        for mod, _cap, method, path in self.MOUNTED_WITHOUT_ADVERTISING
                        if mod == module
                    }
                    self.assertEqual(
                        (mounted - available) - allowed,
                        set(),
                        f"{base} mounts a route it does not advertise, "
                        "and the reason is not written down",
                    )

    def test_declaring_only_tts_leaves_the_design_routes_unreachable(self):
        """The whitelist above, exercised rather than asserted about.

        The three TTS slots share one weight, so declaring supports_tts
        mounts the design routes too. What keeps them out of reach is the
        spec, and nothing else -- so if this ever stops holding, the
        gateway starts routing a capability nobody declared.
        """
        for base, module in (("firered", "firered"), ("breeze", "breeze")):
            with self.subTest(base=base):
                app = self._build(base, ("tts",), module)
                mounted = self._mounted(app)
                available, _described = self._spec(app)

                design = {("POST", "/v1/text-to-voice"), ("POST", "/v1/text-to-voice/design")}
                self.assertTrue(design.issubset(mounted))
                self.assertEqual(design & available, set())
                # And declaring it is what makes them reachable.
                declared = self._build(base, ("tts", "tts_design"), module)
                declared_available, _ = self._spec(declared)
                self.assertTrue(design.issubset(declared_available))


class BreezeAttnFallbackTest(unittest.TestCase):
    def test_flash_rewritten_to_requested(self):
        from wrapper.caps.breeze import _fallback_attn

        self.assertEqual(_fallback_attn("eager", "flash_attention_2"), "eager")
        self.assertEqual(_fallback_attn("sdpa", "flash_attention_3"), "sdpa")
        self.assertEqual(_fallback_attn("eager", "eager"), "eager")
        self.assertEqual(_fallback_attn("eager", None), "eager")
        self.assertEqual(_fallback_attn("", "flash_attention_2"), "eager")


if __name__ == "__main__":
    unittest.main()
