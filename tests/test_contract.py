import glob
import importlib
import importlib.util
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
    # 🔴 Listed here as well as in the catalog, which is the point of this table: a route
    # mounted and not advertised is the easy mistake, and a diagnostic one is the easiest
    # of all because nothing calling the engine misses it.
    ("align", "align", "GET", "/v1/audio/align/telemetry"): {"async_supported": False},
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
    ("whisper_ov", "stt", "POST", "/v1/audio/transcriptions"): {
        "async_supported": True,
    },
    ("whisper_ov", "stt", "POST", "/v1/audio/translations"): {
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
    def test_only_multi_span_transcribers_advertise_segments(self):
        for module in ("stt_stream", "whisper"):
            endpoint = catalog.describe_endpoint(
                module, "stt", "POST", "/v1/audio/transcriptions", True
            )
            self.assertIn(
                {"name": "segments", "type": "array"}, endpoint["parameters"]
            )

        single = catalog.describe_endpoint(
            "future_single_stt", "stt", "POST", "/v1/audio/transcriptions", True
        )
        self.assertNotIn("segments", {p["name"] for p in single["parameters"]})

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
                "watchdog": "qwen-asr",
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
            "whisper_ov": {
                "supports": ["stt"],
                "watchdog": "whisper-ov",
                "state": {
                    "ready": False,
                    "error": None,
                    "model": None,
                    "pipeline": None,
                    "device": None,
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
                if name not in ("stt_stream", "align"):
                    self.assertIs(args[1], module._load)
                    self.assertIs(args[2], module.build_app)
                elif name == "align":
                    with (
                        mock.patch.object(module, "_load", side_effect=RuntimeError("load failed")),
                        mock.patch.object(module, "_p"),
                        mock.patch.object(module.log, "exception"),
                    ):
                        args[1]()
                    self.assertEqual(module._state["error"], "load failed")
                    self.assertFalse(module._state["ready"])
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


class OpenVINOModeTest(unittest.TestCase):
    """Intel is an install-time choice (AUDIO_BASE=ov + OLARES_GPU_MODE), never a CUDA fallback."""

    def test_qwen_base_is_not_openvino(self):
        from wrapper.caps import stt_stream as q

        with mock.patch.dict(os.environ, {"AUDIO_BASE": "qwen"}, clear=False):
            self.assertFalse(q._is_ov())

    def test_ov_base_is_openvino(self):
        from wrapper.caps import stt_stream as q

        with mock.patch.dict(os.environ, {"AUDIO_BASE": "ov"}, clear=False):
            self.assertTrue(q._is_ov())

    def test_intel_modes_select_gpu_device_without_probing_cuda(self):
        from wrapper.caps import stt_stream as q

        with mock.patch.object(q, "OV_DEVICE", ""):
            with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "intel",
                                              "REQUIRED_GPU_MEMORY": "0"}, clear=False):
                self.assertEqual(q._ov_device(), "GPU")
            with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "intel-gpu",
                                              "REQUIRED_GPU_MEMORY": "12Gi"}, clear=False):
                self.assertEqual(q._ov_device(), "GPU")
            with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "nvidia",
                                              "REQUIRED_GPU_MEMORY": "0"}, clear=False):
                self.assertEqual(q._ov_device(), "CPU")

    def test_whisperov_generate_sends_a_float_list(self):
        import types

        from wrapper.caps import whisper_ov as w

        class _Tensor:
            def __init__(self, xs):
                self._xs = list(xs)

            def detach(self):
                return self

            def cpu(self):
                return self

            def float(self):
                return self

            def reshape(self, *_):
                return self

            def __iter__(self):
                return iter(self._xs)

        seen = {}

        def fake_generate(raw, **kw):
            seen["raw"] = raw
            seen["kw"] = kw
            return types.SimpleNamespace(texts=["hello"], text="hello")

        w._state["pipeline"] = types.SimpleNamespace(generate=fake_generate)
        out = w._generate(_Tensor([0.0, 0.25, -0.5]), "transcribe", "en")
        self.assertEqual(out, "hello")
        self.assertIsInstance(seen["raw"], list)
        self.assertEqual(seen["raw"], [0.0, 0.25, -0.5])
        self.assertTrue(all(isinstance(x, float) for x in seen["raw"]))
        self.assertEqual(seen["kw"].get("task"), "transcribe")
        self.assertEqual(seen["kw"].get("language"), "en")
        self.assertNotIn("num_beams", seen["kw"])

    def test_whisper_ir_requires_beam_idx(self):
        from wrapper import ct2_whisper

        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "openvino_encoder_model.xml"), "w").write("<net/>")
            dec = os.path.join(td, "openvino_decoder_model.xml")
            open(dec, "w").write("<net><layer name=\"input_ids\"/></net>")
            self.assertFalse(ct2_whisper.is_whisper_ir(td))
            open(dec, "w").write("<net><layer name=\"beam_idx\"/></net>")
            self.assertTrue(ct2_whisper.is_whisper_ir(td))

    def test_whisperov_refuses_cpu(self):
        from wrapper.caps import whisper_ov as w

        with mock.patch("wrapper.ovutil.device", return_value="CPU"):
            with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "intel"}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "must use GPU"):
                    w._require_gpu()
        with mock.patch("wrapper.ovutil.device", return_value="GPU"):
            with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "intel-gpu"}, clear=False):
                self.assertEqual(w._require_gpu(), "GPU")

    def test_enhanceov_requires_gpu(self):
        from wrapper.caps import enhance

        with mock.patch("wrapper.ovutil.device", return_value="CPU"):
            with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "intel"}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "must use GPU"):
                    enhance._require_ov_gpu()
        with mock.patch("wrapper.ovutil.device", return_value="GPU"):
            with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "intel-gpu"}, clear=False):
                self.assertEqual(enhance._require_ov_gpu(), "GPU")
        with mock.patch.dict(os.environ, {"AUDIO_BASE": "pyannote"}, clear=False):
            class CpuOnly:
                class cuda:
                    @staticmethod
                    def is_available():
                        return False

            self.assertEqual(enhance._speechbrain_device(CpuOnly), "cpu")

    def test_enhance_mask_follows_stft_time_freq(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch")
        from wrapper.caps import enhance

        spec = torch.zeros(1, 11, 257, 2)
        aligned = enhance._align_mask(torch.rand(1, 257, 11), spec)
        self.assertEqual(tuple(aligned.shape), (1, 11, 257, 1))
        aligned = enhance._align_mask(torch.rand(1, 11, 257), spec)
        self.assertEqual(tuple(aligned.shape), (1, 11, 257, 1))

    def test_ct2_whisper_never_fetches_a_second_repo(self):
        import inspect
        import tempfile

        from wrapper import ct2_whisper

        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "model.bin"), "wb").close()
            with open(os.path.join(td, "config.json"), "w", encoding="utf-8") as fh:
                fh.write("{}")
            self.assertTrue(ct2_whisper.is_ct2(td))
            self.assertFalse(ct2_whisper.is_transformers(td))
            src = inspect.getsource(ct2_whisper)
            self.assertNotIn("snapshot_download", src)
            self.assertNotIn("huggingface_hub", src)
            with self.assertRaises((RuntimeError, ImportError)):
                ct2_whisper.to_transformers_dir(td, os.path.join(td, "hf-from-ct2"))
            self.assertNotIn("import ctranslate2", src)
            self.assertIn("_read_ct2_bin", src)

    def test_ct2_whisper_reads_model_bin_without_libctranslate2(self):
        import struct

        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy")
        from wrapper import ct2_whisper

        def write_string(fh, s):
            b = s.encode("utf-8") + b"\0"
            fh.write(struct.pack("<H", len(b)))
            fh.write(b)

        dtypes = {np.dtype("float32"): 0, np.dtype("int8"): 1}
        tensors = {
            "encoder/conv1/weight": np.ones((2, 1, 3), np.float32),
            "encoder/conv1/bias": np.zeros((2,), np.float32),
            "encoder/conv2/weight": np.ones((2, 2, 3), np.float32),
            "encoder/conv2/bias": np.zeros((2,), np.float32),
            "encoder/layer_norm/gamma": np.ones((2,), np.float32),
            "encoder/layer_norm/beta": np.zeros((2,), np.float32),
            "decoder/embeddings": np.ones((4, 2), np.float32),
            "decoder/projection/weight": np.ones((4, 2), np.float32),
            "encoder/layer_0/self_attention/linear_layers/0/weight": np.array(
                [[10, -10], [20, 0]], np.int8),
            "encoder/layer_0/self_attention/linear_layers/0/weight_scale": np.array(
                [2.0, 4.0], np.float32),
        }
        with tempfile.TemporaryDirectory() as td:
            bin_path = os.path.join(td, "model.bin")
            with open(bin_path, "wb") as fh:
                fh.write(struct.pack("<I", 6))
                write_string(fh, "WhisperSpec")
                fh.write(struct.pack("<I", 1))
                fh.write(struct.pack("<I", len(tensors)))
                for name, arr in tensors.items():
                    arr = np.ascontiguousarray(arr)
                    write_string(fh, name)
                    fh.write(struct.pack("B", arr.ndim))
                    for dim in arr.shape:
                        fh.write(struct.pack("<I", int(dim)))
                    fh.write(struct.pack("B", dtypes[arr.dtype]))
                    fh.write(struct.pack("<I", arr.nbytes))
                    fh.write(arr.tobytes())
                fh.write(struct.pack("<I", 0))
            with open(os.path.join(td, "config.json"), "w", encoding="utf-8") as fh:
                json.dump({"encoder_layers": 1, "decoder_layers": 0}, fh)
            dumped = ct2_whisper._dump_ct2_state_dict(td)
            q = dumped["encoder/layer_0/self_attention/linear_layers/0/weight"]
            self.assertEqual(q.dtype, np.float32)
            self.assertTrue(np.allclose(q, np.array([[5.0, -5.0], [5.0, 0.0]], np.float32)))
            mapped = ct2_whisper._to_hf_names(dumped, {"encoder_layers": 1, "decoder_layers": 0})
            self.assertGreaterEqual(len(mapped), 8)
            self.assertEqual(mapped["model.encoder.conv1.weight"].shape, (2, 1, 3))
            self.assertTrue(np.allclose(
                mapped["model.encoder.layers.0.self_attn.q_proj.weight"], q))
            hf = ct2_whisper._whisper_hf_config(dumped)
            self.assertEqual(hf["model_type"], "whisper")
            self.assertEqual(hf["encoder_layers"], 1)
            self.assertEqual(hf["num_mel_bins"], 1)

    def test_ct2_unfuses_whisper_attention(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy")
        from wrapper import ct2_whisper

        qkv = np.arange(12, dtype=np.float32).reshape(6, 2)
        out_w = np.ones((2, 2), np.float32)
        qkv_b = np.array([1, 2, 3, 4, 5, 6], np.float32)
        kv = np.arange(8, dtype=np.float32).reshape(4, 2)
        state = {
            "encoder/conv1/weight": np.ones((2, 1, 3), np.float32),
            "encoder/conv1/bias": np.zeros((2,), np.float32),
            "encoder/conv2/weight": np.ones((2, 2, 3), np.float32),
            "encoder/conv2/bias": np.zeros((2,), np.float32),
            "encoder/layer_norm/gamma": np.ones((2,), np.float32),
            "encoder/layer_norm/beta": np.zeros((2,), np.float32),
            "encoder/layer_0/self_attention/linear_0/weight": qkv,
            "encoder/layer_0/self_attention/linear_0/bias": qkv_b,
            "encoder/layer_0/self_attention/linear_1/weight": out_w,
            "decoder/embeddings": np.ones((4, 2), np.float32),
            "decoder/layer_0/attention/linear_0/weight": np.eye(2, dtype=np.float32),
            "decoder/layer_0/attention/linear_1/weight": kv,
            "decoder/layer_0/attention/linear_2/weight": out_w,
        }
        mapped = ct2_whisper._to_hf_names(state, {"encoder_layers": 1, "decoder_layers": 1})
        self.assertEqual(mapped["model.encoder.layers.0.self_attn.q_proj.weight"].tolist(),
                         qkv[:2].tolist())
        self.assertEqual(mapped["model.encoder.layers.0.self_attn.k_proj.weight"].tolist(),
                         qkv[2:4].tolist())
        self.assertEqual(mapped["model.encoder.layers.0.self_attn.v_proj.weight"].tolist(),
                         qkv[4:].tolist())
        self.assertEqual(mapped["model.encoder.layers.0.self_attn.q_proj.bias"].tolist(),
                         [1, 2])
        self.assertEqual(mapped["model.encoder.layers.0.self_attn.out_proj.weight"].tolist(),
                         out_w.tolist())
        self.assertEqual(mapped["model.decoder.layers.0.encoder_attn.k_proj.weight"].tolist(),
                         kv[:2].tolist())
        self.assertEqual(mapped["model.decoder.layers.0.encoder_attn.v_proj.weight"].tolist(),
                         kv[2:].tolist())
        self.assertEqual(mapped["proj_out.weight"].shape, (4, 2))

    def test_ct2_dest_without_model_type_is_repaired(self):
        from wrapper import ct2_whisper

        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src")
            dest = os.path.join(td, "dest")
            os.makedirs(src)
            os.makedirs(dest)
            open(os.path.join(src, "model.bin"), "wb").close()
            with open(os.path.join(src, "config.json"), "w", encoding="utf-8") as fh:
                json.dump({"alignment_heads": []}, fh)
            open(os.path.join(dest, "model.safetensors"), "wb").close()
            with open(os.path.join(dest, "config.json"), "w", encoding="utf-8") as fh:
                json.dump({"alignment_heads": []}, fh)
            self.assertFalse(ct2_whisper._hf_config_ok(dest))
            out = ct2_whisper.to_transformers_dir(src, dest)
            self.assertEqual(out, dest)
            self.assertTrue(ct2_whisper._hf_config_ok(dest))
            with open(os.path.join(dest, "config.json"), encoding="utf-8") as fh:
                cfg = json.load(fh)
            self.assertEqual(cfg["architectures"], ["WhisperForConditionalGeneration"])
            self.assertTrue(os.path.isfile(os.path.join(dest, "preprocessor_config.json")))
            with open(os.path.join(dest, "generation_config.json"), encoding="utf-8") as fh:
                gen = json.load(fh)
            self.assertEqual(gen["lang_to_id"]["<|en|>"], 50259)
            self.assertEqual(gen["task_to_id"]["transcribe"], 50360)

    def test_ct2_dest_missing_lang_to_id_is_repaired_into_openvino(self):
        from wrapper import ct2_whisper

        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "src")
            dest = os.path.join(td, "dest")
            nested = os.path.join(dest, "openvino")
            os.makedirs(src)
            os.makedirs(nested)
            open(os.path.join(src, "model.bin"), "wb").close()
            with open(os.path.join(src, "config.json"), "w", encoding="utf-8") as fh:
                json.dump({"alignment_heads": []}, fh)
            open(os.path.join(dest, "model.safetensors"), "wb").close()
            with open(os.path.join(dest, "config.json"), "w", encoding="utf-8") as fh:
                json.dump({
                    "model_type": "whisper",
                    "architectures": ["WhisperForConditionalGeneration"],
                }, fh)
            open(os.path.join(nested, "openvino_encoder_model.xml"), "wb").close()
            open(os.path.join(nested, "openvino_decoder_model.xml"), "wb").close()
            self.assertFalse(ct2_whisper._hf_config_ok(dest))
            out = ct2_whisper.to_transformers_dir(src, dest)
            self.assertEqual(out, dest)
            self.assertTrue(ct2_whisper._hf_config_ok(dest))
            with open(os.path.join(nested, "generation_config.json"), encoding="utf-8") as fh:
                gen = json.load(fh)
            self.assertIn("<|zh|>", gen["lang_to_id"])

    def test_device_flag_overrides_mode(self):
        from wrapper.caps import stt_stream as q

        with mock.patch.object(q, "OV_DEVICE", "GPU.1"):
            with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "intel"}, clear=False):
                self.assertEqual(q._ov_device(), "GPU.1")

    def test_absent_max_model_len_follows_install_mode(self):
        from wrapper.caps import stt_stream as q

        with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "nvidia"}, clear=False):
            self.assertEqual(q._default_max_model_len(), 8192)
        with mock.patch.dict(os.environ, {"OLARES_GPU_MODE": "nvidia-gb10"}, clear=False):
            self.assertEqual(q._default_max_model_len(), 3072)

    def test_absent_enforce_eager_is_on_for_nvidia_off_for_ov(self):
        from wrapper.caps import stt_stream as q

        with mock.patch.dict(os.environ, {"AUDIO_BASE": "qwen"}, clear=False):
            self.assertTrue(q._default_enforce_eager())
        with mock.patch.dict(os.environ, {"AUDIO_BASE": "ov"}, clear=False):
            self.assertFalse(q._default_enforce_eager())

    def test_written_max_model_len_and_eager_keep_the_user_value(self):
        from wrapper.contract import EngineArgs

        args = EngineArgs("--max-model-len 4096 --enforce-eager false")
        self.assertEqual(args.count("--max-model-len", 8192), 4096)
        self.assertFalse(args.switch("--enforce-eager", True))

    def test_ov_generate_many_sends_the_group_in_one_call(self):
        import types
        from wrapper.caps import stt_stream as q

        seen = []

        def fake_generate(raw, **kw):
            seen.append(raw)
            return types.SimpleNamespace(texts=["one", "two"])

        class _Clip:
            def __init__(self, n):
                self._n = n

            def astype(self, _dt):
                return self

            def reshape(self, *a, **k):
                return self

            def tolist(self):
                return [0.0] * self._n

        asr = q._state.get("asr")
        try:
            q._state["asr"] = types.SimpleNamespace(generate=fake_generate)
            out = q._ov_generate_many([_Clip(16000), _Clip(32000)])
            self.assertEqual(out, ["one", "two"])
            self.assertEqual(len(seen), 1)
            self.assertEqual(len(seen[0]), 2)
            self.assertEqual(len(seen[0][0]), 16000)
            self.assertEqual(len(seen[0][1]), 32000)
        finally:
            q._state["asr"] = asr

    def test_ov_batch_probe_fails_load_when_generate_rejects_a_list(self):
        import types
        from wrapper.caps import stt_stream as q

        was = q.MAX_BATCH_SPANS
        try:
            q.MAX_BATCH_SPANS = 8

            def reject(raw, **kw):
                if raw and isinstance(raw[0], list):
                    raise TypeError("AudioInputs")
                return types.SimpleNamespace(texts=["ok"])

            with self.assertRaises(RuntimeError) as ctx:
                q._ov_assert_batch_generate(types.SimpleNamespace(generate=reject))
            self.assertIn("rejects a list of waveforms", str(ctx.exception))

            def echo(raw, **kw):
                return types.SimpleNamespace(texts=["And so my.", "And so my."])

            with self.assertRaises(RuntimeError) as ctx:
                q._ov_assert_batch_generate(types.SimpleNamespace(generate=echo))
            self.assertIn("copied one text", str(ctx.exception))

            def boom(raw, **kw):
                raise RuntimeError("[GPU] CL_OUT_OF_RESOURCES exception.")

            with self.assertRaises(RuntimeError) as ctx:
                q._ov_assert_batch_generate(types.SimpleNamespace(generate=boom))
            self.assertIn("died on GPU", str(ctx.exception))
        finally:
            q.MAX_BATCH_SPANS = was

    def test_ov_qwen3_batch_patch_stacks_decoder_and_reshapes_encoder_batch(self):
        spec = importlib.util.spec_from_file_location(
            "apply_qwen3_asr_batch",
            os.path.join(os.path.dirname(__file__),
                         "../bases/ov/patches/apply_qwen3_asr_batch.py"),
        )
        patch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch)

        hpp = "using AudioInputs = std::variant<std::vector<float>>;\n"
        cpp = (
            "#include \"pipeline.hpp\"\n"
            "#include <algorithm>\n"
            "namespace ov::genai {\n"
            "    const std::vector<float>& audio = std::visit(\n"
            "        ov::genai::utils::overloaded{\n"
            "            [](const std::vector<float>& input) -> const std::vector<float>& {\n"
            "                return input;\n"
            "            },\n"
            "        },\n"
            "        audio_inputs);\n"
            "    const std::vector<AudioChunk> chunks =\n"
            "        split_audio_into_chunks({audio}, m_feature_extractor.sampling_rate, MAX_ASR_INPUT_SECONDS);\n"
            "    for (size_t batch = 0; batch < batch_size; ++batch) {\n"
            "        const auto text = m_tokenizer.decode(encoded_results.tokens[0]);\n"
            "        results.push_back(text);\n"
            "    }\n"
            "    return results;\n"
            "}\n"
        )
        decoder = (
            "#include \"decoder.hpp\"\n"
            "namespace ov::genai {\n"
            "    ov::CompiledModel compiled_model =\n"
            "        core.compile_model(models_path / \"openvino_decoder_model.xml\", device, properties);\n"
            "    m_request.reset_state();\n"
            "    auto free_finished_requests = [&active_sequence_groups]() {\n"
            "        auto removed_it =\n"
            "            std::remove_if(active_sequence_groups.begin(),\n"
            "                           active_sequence_groups.end(),\n"
            "                           [](const SequenceGroup::Ptr& sg) {\n"
            "                               return sg->has_finished() || sg->handle_stopped() || sg->handle_cancelled();\n"
            "                           });\n"
            "        active_sequence_groups.erase(removed_it, active_sequence_groups.end());\n"
            "    };\n"
            "    while (!active_sequence_groups.empty()) {\n"
            "        ov::Tensor new_input_ids(ov::element::i64, {total_num_tokens, 1});\n"
            "        m_request.set_tensor(\"input_ids\", new_input_ids);\n"
            "        m_request.set_tensor(\"beam_idx\", ov::Tensor{ov::element::i32, {total_num_tokens}, next_beams.data()});\n"
            "    }\n"
            "}\n"
        )
        encoder = (
            "ov::Tensor Qwen3ASREncoder::encode(const WhisperFeatures& features) {\n"
            "    ov::Tensor input_tensor = chunk_mel_features(features);\n"
            "    m_request.set_tensor(\"input_features\", input_tensor);\n"
            "    m_request.set_tensor(\"input_features\", ov::Tensor(ov::element::f32, {0, 0, 0}));\n"
            "    return output;\n"
            "}\n"
        )
        other = (
            "            [](const std::vector<float>& input) -> const std::vector<float>& {\n"
            "                return input;\n"
            "            },\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "src")
            asr = os.path.join(root, "automatic_speech_recognition")
            os.makedirs(os.path.join(asr, "models", "qwen3-asr"))
            os.makedirs(os.path.join(asr, "models", "whisper"))
            open(os.path.join(asr, "pipeline.hpp"), "w").write(hpp)
            open(os.path.join(asr, "models", "qwen3-asr", "pipeline.cpp"), "w").write(cpp)
            open(os.path.join(asr, "models", "qwen3-asr", "decoder.cpp"), "w").write(decoder)
            open(os.path.join(asr, "models", "qwen3-asr", "encoder.cpp"), "w").write(encoder)
            whisper = os.path.join(asr, "models", "whisper", "pipeline.cpp")
            open(whisper, "w").write(other)
            old = sys.argv
            try:
                sys.argv = ["apply_qwen3_asr_batch.py", tmp]
                patch.main()
            finally:
                sys.argv = old
            got_hpp = open(os.path.join(asr, "pipeline.hpp")).read()
            got_cpp = open(os.path.join(asr, "models", "qwen3-asr", "pipeline.cpp")).read()
            got_dec = open(os.path.join(asr, "models", "qwen3-asr", "decoder.cpp")).read()
            got_enc = open(os.path.join(asr, "models", "qwen3-asr", "encoder.cpp")).read()
            got_wh = open(whisper).read()
        self.assertIn("std::vector<std::vector<float>>", got_hpp)
        self.assertIn("split_audio_into_chunks(audios,", got_cpp)
        self.assertIn("stack_encoder_hiddens", got_cpp)
        self.assertIn("tokens[i]", got_cpp)
        self.assertIn("Dimension::dynamic()", got_dec)
        self.assertIn("encoder_hidden_states", got_dec)
        self.assertIn("keep_encoder_hidden_batch", got_dec)
        self.assertIn("encoder flatten Unsqueeze", got_dec)
        self.assertIn("min_intel_gpu_audio_samples", got_cpp)
        self.assertIn("min_intel_gpu_encoder_frames", got_cpp)
        self.assertIn("stacked_t", got_cpp)
        self.assertIn("audio_token_counts", got_cpp)
        self.assertIn("create_infer_request()", got_enc)
        self.assertNotIn("{0, 0, 0}", got_enc)
        self.assertIn("cannot realloc", got_dec)
        self.assertIn("{batch_size, 1}", got_dec)
        self.assertNotIn("fix_encoder_gather_batch", got_dec)
        self.assertNotIn("batch*T", got_dec)
        self.assertIn("batched audio is only implemented for Qwen3-ASR", got_wh)

    def test_ov_base_implements_stt_and_align(self):
        self.assertEqual(catalog.implements("ov"), ["stt", "stt_stream", "align"])
        self.assertEqual(catalog.module_of("ov", "align"), "align")
        self.assertEqual(catalog.module_of("ov", "stt_stream"), "stt_stream")

    def test_whisperov_and_enhanceov_are_their_own_bases(self):
        self.assertEqual(catalog.implements("whisperov"), ["stt"])
        self.assertEqual(catalog.module_of("whisperov", "stt"), "whisper_ov")
        self.assertIsNone(catalog.module_of("whisperov", "stt_stream"))
        self.assertEqual(catalog.implements("enhanceov"), ["enhance"])
        self.assertEqual(catalog.module_of("enhanceov", "enhance"), "enhance")
        self.assertNotEqual(catalog.module_of("ov", "stt"), "whisper_ov")
        self.assertEqual(catalog.implements("breezeov"), ["tts", "tts_clone", "tts_design"])
        self.assertEqual(catalog.module_of("breezeov", "tts"), "breeze")
        self.assertEqual(catalog.implements("fireredov"), ["tts", "tts_clone", "tts_design"])
        self.assertEqual(catalog.module_of("fireredov", "tts"), "firered")

    def test_ov_engine_spec_is_v2_like_main(self):
        with mock.patch.dict(os.environ, {"AUDIO_BASE": "ov",
                                          "MODEL_SUPPORTS": "supports_align"}, clear=False):
            app = FastAPI()
            contract.register(app, model_name="Qwen/Qwen3-ForcedAligner-0.6B",
                              module="align", served=["align"], is_ready=lambda: True)
            spec = TestClient(app).get("/api/engine-spec").json()
        self.assertEqual(spec["schema_version"], 2)
        self.assertEqual(spec["base"], "qwen")
        self.assertEqual(spec["model"], "Qwen/Qwen3-ForcedAligner-0.6B")
        self.assertIsInstance(spec["implements"], list)
        self.assertEqual(spec["serves"], ["align"])
        self.assertEqual(spec["declares"], ["align"])
        self.assertTrue(spec["endpoints"])
        by_cap = {e.get("capability"): e for e in spec["endpoints"] if e.get("capability")}
        self.assertTrue(by_cap["align"]["available"])
        self.assertFalse(by_cap["stt"]["available"])
        self.assertFalse(by_cap["stt_stream"]["available"])

    def test_align_ov_export_is_one_shot_asr_not_hf(self):
        from wrapper.caps import align as a

        cmd = a._ov_export_cmd("/src", "/dest")
        self.assertEqual(cmd[cmd.index("--task") + 1], "automatic-speech-recognition")
        self.assertNotIn("token-classification", cmd)
        self.assertNotIn("with-past", "".join(cmd))

    def test_align_ov_ir_accepts_one_shot_asr_without_marker(self):
        from wrapper.caps import align as a

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "openvino_encoder_model.xml"), "w") as f:
                f.write("<net/>")
            with open(os.path.join(td, "openvino_decoder_model.xml"), "w") as f:
                f.write("<net><layer name=\"input_ids\"/></net>")
            self.assertTrue(a._looks_like_ov_ir(td))
            with open(os.path.join(td, "openvino_decoder_model.xml"), "w") as f:
                f.write("<net><layer name=\"beam_idx\"/><layer name=\"input_ids\"/></net>")
            self.assertFalse(a._looks_like_ov_ir(td))

    def test_ov_logits_splits_encoder_decoder_and_skips_thinker(self):
        from wrapper.caps import align as a

        class Enc:
            def __call__(self, **kw):
                return mock.Mock(last_hidden_state="h")

        class Dec:
            def __call__(self, **kw):
                if kw.get("encoder_hidden_states") == "h" and kw.get("input_ids") == 1:
                    return mock.Mock(logits="ok")
                raise TypeError("bad decoder args %s" % kw)

        class Fake:
            def __init__(self):
                self.encoder = Enc()
                self.decoder = Dec()
                self.thinker = mock.Mock(side_effect=TypeError(
                    "OVModelForSeq2SeqLM.forward() got multiple values for "
                    "keyword argument 'input_ids'"))

            def __call__(self, **kw):
                raise TypeError("must not call SpeechSeq2Seq.forward")

        self.assertEqual(a._ov_logits(Fake(), {"input_ids": 1, "input_features": 2}), "ok")

    def test_nvidia_mode_does_not_flip_cuda_metrics_to_dri(self):
        text = gpu.gpu_metrics_text()
        self.assertIn("gpu_present 0", text)

    def test_looks_like_ov_ir_rejects_stateless_whisper_decoder(self):
        from wrapper.caps import stt_stream as q

        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(q._looks_like_ov_ir(td))
            with open(os.path.join(td, "openvino_encoder_model.xml"), "w") as f:
                f.write("<net/>")
            with open(os.path.join(td, "openvino_decoder_model.xml"), "w") as f:
                f.write("<net><layer name=\"input_ids\"/></net>")
            self.assertFalse(q._looks_like_ov_ir(td))
            with open(os.path.join(td, "openvino_decoder_model.xml"), "w") as f:
                f.write("<net><layer name=\"beam_idx\"/><layer name=\"input_ids\"/></net>")
            self.assertTrue(q._looks_like_ov_ir(td))

    def test_ov_export_cmd_uses_with_past(self):
        from wrapper.caps import stt_stream as q

        cmd = q._ov_export_cmd("/src", "/dest")
        self.assertEqual(cmd[cmd.index("--task") + 1],
                         "automatic-speech-recognition-with-past")
        self.assertNotIn("automatic-speech-recognition", cmd)

    def test_ensure_ov_ir_drops_stateless_cache_and_reexports(self):
        from wrapper.caps import stt_stream as q

        with tempfile.TemporaryDirectory() as src:
            nested = os.path.join(src, "openvino")
            os.makedirs(nested)
            with open(os.path.join(nested, "openvino_encoder_model.xml"), "w") as f:
                f.write("<net/>")
            with open(os.path.join(nested, "openvino_decoder_model.xml"), "w") as f:
                f.write("<net><layer name=\"input_ids\"/></net>")

            def fake_export(cmd, *a, **k):
                dest = cmd[-1]
                os.makedirs(dest, exist_ok=True)
                with open(os.path.join(dest, "openvino_encoder_model.xml"), "w") as f:
                    f.write("<net/>")
                with open(os.path.join(dest, "openvino_decoder_model.xml"), "w") as f:
                    f.write("<net><layer name=\"beam_idx\"/></net>")

            with mock.patch("subprocess.check_call", side_effect=fake_export) as cc:
                out = q._ensure_ov_ir(src)
            self.assertEqual(out, nested)
            argv = cc.call_args[0][0]
            self.assertEqual(argv[argv.index("--task") + 1],
                             "automatic-speech-recognition-with-past")
            self.assertTrue(q._looks_like_ov_ir(out))

    def test_ensure_ov_ir_skips_export_when_beam_idx_ir_exists(self):
        from wrapper.caps import stt_stream as q

        with tempfile.TemporaryDirectory() as src:
            nested = os.path.join(src, "openvino")
            os.makedirs(nested)
            with open(os.path.join(nested, "openvino_encoder_model.xml"), "w") as f:
                f.write("<net/>")
            with open(os.path.join(nested, "openvino_decoder_model.xml"), "w") as f:
                f.write("<net><layer name=\"beam_idx\"/></net>")
            with mock.patch("subprocess.check_call") as cc:
                self.assertEqual(q._ensure_ov_ir(src), nested)
            cc.assert_not_called()

    def test_ov_warmup_failure_fails_load(self):
        from wrapper.caps import stt_stream as q

        with mock.patch.object(q, "_is_ov", return_value=True):
            with mock.patch.object(q, "_offline_transcribe", side_effect=RuntimeError("beam_idx")):
                with self.assertRaises(RuntimeError) as ctx:
                    q._warmup()
        self.assertIn("openvino warmup failed", str(ctx.exception))

    def test_vllm_warmup_failure_does_not_block_ready(self):
        from wrapper.caps import stt_stream as q

        with mock.patch.object(q, "_is_ov", return_value=False):
            with mock.patch.object(q, "_offline_transcribe", side_effect=RuntimeError("cold")):
                q._warmup()

    def test_ov_repetition_fallback_uses_the_same_per_second_cap(self):
        from wrapper.caps import stt_stream as q

        was_on, was_n, was_said = q.REPETITION_ON, q.REPETITION_FALLBACK_TOKENS_PER_SEC, q._repdet_said[:]
        try:
            q.REPETITION_ON = False
            q.REPETITION_FALLBACK_TOKENS_PER_SEC = 12
            with mock.patch.object(q, "_is_ov", return_value=True):
                self.assertEqual(q._ov_max_new_tokens(10.0), q.OV_MAX_NEW_TOKENS)
                q.REPETITION_ON = True
                want = min(q.OFFLINE_MAX_TOKENS, int(10.0 * 12) + q.TOKENS_FLOOR)
                self.assertEqual(q._ov_max_new_tokens(10.0), want)
                q.REPETITION_FALLBACK_TOKENS_PER_SEC = 0
                self.assertEqual(q._ov_max_new_tokens(10.0), q.OV_MAX_NEW_TOKENS)
        finally:
            q.REPETITION_ON = was_on
            q.REPETITION_FALLBACK_TOKENS_PER_SEC = was_n
            del q._repdet_said[:]
            q._repdet_said.extend(was_said)

    def test_ov_generate_passes_the_fallback_budget(self):
        import types
        from wrapper.caps import stt_stream as q

        seen = []

        def fake_generate(raw, **kw):
            seen.append(kw.get("max_new_tokens"))
            return types.SimpleNamespace(texts=["ok"])

        was_on, was_n, was_said = q.REPETITION_ON, q.REPETITION_FALLBACK_TOKENS_PER_SEC, q._repdet_said[:]
        asr = q._state.get("asr")
        try:
            q.REPETITION_ON = True
            q.REPETITION_FALLBACK_TOKENS_PER_SEC = 12
            q._state["asr"] = types.SimpleNamespace(generate=fake_generate)

            class _Arr:
                def astype(self, _dt):
                    return self

                def reshape(self, *a, **k):
                    return self

                def tolist(self):
                    return [0.0] * 16000

            with mock.patch.object(q, "_is_ov", return_value=True):
                q._ov_generate(_Arr())
            self.assertEqual(seen, [min(q.OFFLINE_MAX_TOKENS, 12 + q.TOKENS_FLOOR)])
        finally:
            q.REPETITION_ON = was_on
            q.REPETITION_FALLBACK_TOKENS_PER_SEC = was_n
            del q._repdet_said[:]
            q._repdet_said.extend(was_said)
            q._state["asr"] = asr

    def test_ov_repetition_report_does_not_look_for_sampling_params(self):
        from wrapper.caps import stt_stream as q

        lines = []
        was_on, was_n, was_said = q.REPETITION_ON, q.REPETITION_FALLBACK_TOKENS_PER_SEC, q._repdet_said[:]
        try:
            q.REPETITION_ON = True
            q.REPETITION_FALLBACK_TOKENS_PER_SEC = 12
            del q._repdet_said[:]
            with mock.patch.object(q, "_is_ov", return_value=True):
                with mock.patch.object(q, "_p", side_effect=lines.append):
                    q._say_repetition_once()
            self.assertTrue(any("OpenVINO has no vLLM detector" in x for x in lines), lines)
            self.assertFalse(any("sampling_params" in x for x in lines), lines)
        finally:
            q.REPETITION_ON = was_on
            q.REPETITION_FALLBACK_TOKENS_PER_SEC = was_n
            del q._repdet_said[:]
            q._repdet_said.extend(was_said)


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


class EngineArgsTest(unittest.TestCase):
    """What happens to a flag whose value cannot be read.

    Falling back to the default is right -- an engine that refuses to start over a typo is
    worse. The problem is that the fallback is invisible: a deployment that asked for
    batching and mistyped the number gets the serial path, running correctly and slowly,
    and nothing in the logs separates that from never having asked.
    """

    def _args(self, raw):
        from wrapper.contract import EngineArgs

        return EngineArgs(raw)

    def test_an_unreadable_number_falls_back_and_says_so(self):
        args = self._args("--batch-max-spans 32x")
        self.assertEqual(args.count("--batch-max-spans", 1), 1)
        log = _CollectingLog()
        args.warn_unclaimed(log)
        self.assertTrue(
            any("--batch-max-spans" in line and "not a number" in line for line in log.lines),
            log.lines,
        )

    def test_a_readable_number_says_nothing(self):
        args = self._args("--batch-max-spans 32")
        self.assertEqual(args.count("--batch-max-spans", 1), 32)
        log = _CollectingLog()
        args.warn_unclaimed(log)
        self.assertEqual(log.lines, [])

    def test_a_flag_given_with_no_value_says_so(self):
        """The likeliest typo of all: --batch-max-spans with the number left off.

        text() answers the default for it, exactly as for a flag nobody passed, so without
        a separate check this is the one mistake that reports nothing -- and the deployment
        that meant to switch batching on runs the serial path instead, correct and slow.
        """
        args = self._args("--batch-max-spans")
        self.assertEqual(args.count("--batch-max-spans", 1), 1)
        log = _CollectingLog()
        args.warn_unclaimed(log)
        self.assertTrue(
            any("--batch-max-spans" in line and "no value" in line for line in log.lines),
            log.lines,
        )

    def test_a_flag_whose_value_is_only_whitespace_says_so_too(self):
        """`--flag=` is the same mistake as `--flag`, and has to report the same way.

        Fixing one spelling and not the other is how the first round of this went: the
        warning existed and the likeliest typo still slipped past it.
        """
        for raw in ("--batch-max-spans=", "--batch-max-spans= ", "--batch-max-spans"):
            args = self._args(raw)
            self.assertEqual(args.count("--batch-max-spans", 1), 1, raw)
            log = _CollectingLog()
            args.warn_unclaimed(log)
            self.assertTrue(any("--batch-max-spans" in line for line in log.lines),
                            (raw, log.lines))

    def test_a_switch_with_a_stray_equals_sign_still_means_on(self):
        """`--flag=` is the bare flag with a typo, and must not come back meaning the opposite.

        A chart writing `--flag={{ .Values.x }}` with x unset renders exactly this. Reading
        it as off turns the feature off while its author reads the template as turning it on,
        and nothing anywhere says which of the two happened.
        """
        for raw in ("--repetition-detection", "--repetition-detection=",
                    "--repetition-detection= ", '--repetition-detection ""'):
            self.assertTrue(self._args(raw).switch("--repetition-detection"), raw)
        for raw in ("--repetition-detection false", "--repetition-detection 0", ""):
            self.assertFalse(self._args(raw).switch("--repetition-detection"), raw)

    def test_the_warning_does_not_quote_a_value_nobody_typed(self):
        args = self._args("--batch-max-spans")
        args.count("--batch-max-spans", 1)
        log = _CollectingLog()
        args.warn_unclaimed(log)
        self.assertTrue(any("given with no value" in line for line in log.lines), log.lines)
        self.assertFalse(any("<" in line for line in log.lines), log.lines)

    def test_a_switch_value_in_neither_word_list_falls_back_and_says_so(self):
        """🔴 It used to read as OFF in silence, while the same typo on a number-taking flag
        was reported. That asymmetry cost nothing while every switch defaulted off -- a
        misspelling landed on the default -- and stopped costing nothing the day one
        defaulted ON: `--align-batch enable` turned the feature off and the log said nothing.
        """
        log = _CollectingLog()
        args = contract.EngineArgs("--align-batch enable")
        self.assertIs(args.switch("--align-batch", True), True,
                      "an unrecognised value silently overrode a default of on")
        args.warn_unclaimed(log)
        self.assertTrue(any("--align-batch" in m and "enable" in m for m in log.lines),
                        "nothing was said about a value the parser could not read: %r"
                        % (log.lines,))

    def test_the_words_that_do_read_are_still_silent(self):
        # ⚠️ The other half. A warning on a value that IS understood would train operators
        # to ignore the line, which is how the number readers' warning stays useful.
        for raw, default, want in (("--align-batch off", True, False),
                                   ("--align-batch 0", True, False),
                                   ("--align-batch on", False, True),
                                   ("--align-batch", False, True),
                                   ("--align-batch=", False, True)):
            log = _CollectingLog()
            args = contract.EngineArgs(raw)
            self.assertIs(args.switch("--align-batch", default), want, raw)
            args.warn_unclaimed(log)
            self.assertEqual([m for m in log.lines if "--align-batch" in m], [],
                             "%r is a value this parser understands, so it must not warn"
                             % raw)

    def test_given_separates_a_valueless_flag_from_an_absent_one(self):
        self.assertTrue(self._args("--repetition-fallback-tokens-per-sec").given(
            "--repetition-fallback-tokens-per-sec"))
        self.assertTrue(self._args("--repetition-fallback-tokens-per-sec 30").given(
            "--repetition-fallback-tokens-per-sec"))
        self.assertFalse(self._args("").given("--repetition-fallback-tokens-per-sec"))

    def test_an_absent_flag_is_not_an_unreadable_one(self):
        """The default path must stay silent, or the warning becomes noise everyone filters out."""
        args = self._args("")
        self.assertEqual(args.count("--batch-max-spans", 1), 1)
        log = _CollectingLog()
        args.warn_unclaimed(log)
        self.assertEqual(log.lines, [])


class _CollectingLog:
    def __init__(self):
        self.lines = []

    def warning(self, msg, *fmt):
        self.lines.append(msg % fmt if fmt else msg)


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
        # Two reads: the one that fits and the one that crosses. The other 62 MiB never arrive.
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
            {"AUDIO_BASE": "qwen", "MODEL_SUPPORTS": "supports_stt"},
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
        self.assertEqual(spec["declares"], ["stt"])
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
        self.assertIn(
            {"name": "segments", "type": "array"}, transcription["parameters"]
        )
        self.assertTrue(all("max_input_seconds" not in endpoint for endpoint in spec["endpoints"]))

    def test_engine_capacity_reports_the_single_inference_worker(self):
        response = self.client.get("/api/engine-capacity")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"max_concurrency": 1})
        self.assertNotIn("queue", response.json())

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

    # Mounted, and deliberately not advertised as available.
    MOUNTED_WITHOUT_ADVERTISING = frozenset({
        # FireRed / Breeze serve design on these routes but do not advertise tts_design here.
        ("firered", "tts_design", "POST", "/v1/text-to-voice"),
        ("firered", "tts_design", "POST", "/v1/text-to-voice/design"),
        ("breeze", "tts_design", "POST", "/v1/text-to-voice"),
        ("breeze", "tts_design", "POST", "/v1/text-to-voice/design"),
        # Qwen3-TTS reads its preset library off the checkpoint, so voices is checkpoint-gated.
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
                # Starlette adds these for free; the catalog does not advertise them.
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


class NvmlFallbackIsOptIn(unittest.TestCase):
    """NVML answers only for a caller that asked for it.

    The two sources do not measure the same thing: CUDA is intercepted by memory
    virtualization and reports this container's slice, NVML reports the whole card. An
    engine that never asked for the second one must not start reporting it because its
    image gained nvidia-ml-py -- the gauge names and shape stay identical, so nothing
    about that day would look like a change.
    """

    def _gauges(self, text):
        return {line.split()[0]: line.split()[1]
                for line in text.splitlines() if line and not line.startswith("#")}

    def _metrics(self, **kwargs):
        """The body /metrics returns with NVML answering and torch unavailable.

        torch is forced absent rather than left to the host: on a machine that has it with a
        working card, the torch branch answers first and both cases below would pass without
        exercising anything.
        """
        app = FastAPI()
        gpu.mount_metrics(app, **kwargs)
        with mock.patch.dict(sys.modules, {"torch": None}):
            with mock.patch.object(gpu, "_nvml_stats", return_value=(7, 11, 0.5)):
                with TestClient(app) as client:
                    return client.get("/metrics").text

    def test_default_caller_reports_zeros_even_when_nvml_would_answer(self):
        body = self._metrics()
        gauges = self._gauges(body)
        self.assertEqual(gauges["gpu_present"], "0")
        self.assertEqual(gauges["gpu_mem_used_bytes"], "0")
        self.assertEqual(gauges["gpu_mem_total_bytes"], "0")
        self.assertNotIn("as NVML reports it", body)

    def test_opted_in_caller_reports_what_nvml_says(self):
        body = self._metrics(nvml_fallback=True)
        gauges = self._gauges(body)
        self.assertEqual(gauges["gpu_present"], "1")
        self.assertEqual(gauges["gpu_mem_used_bytes"], "7")
        self.assertEqual(gauges["gpu_mem_total_bytes"], "11")
        self.assertIn("as NVML reports it", body)


class FasterWhisperNoCudaTorchRecipeTest(unittest.TestCase):
    def test_family_does_not_install_cuda_torch(self):
        path = os.path.join(os.path.dirname(__file__),
                            "../bases/fasterwhisper/deps.Dockerfile")
        with open(path) as fh:
            text = fh.read()
        self.assertNotIn("FROM ${RUNTIME_IMAGE}", text)
        self.assertNotIn("lovehunter9/audio-runtime", text)
        self.assertNotIn("beclab/audio-runtime", text)
        self.assertNotIn("strip_unused_cuda.sh", text)
        self.assertNotIn("download.pytorch.org/whl/cu128", text)
        self.assertNotIn("download.pytorch.org/whl/cu130", text)
        self.assertIn("download.pytorch.org/whl/cpu", text)
        self.assertIn("nvidia-ml-py", text)
        self.assertIn("nvidia-cublas-cu12", text)
        self.assertIn("nvidia-cudnn-cu12", text)
        self.assertIn("collect_ct2_runtime.py", text)
        self.assertNotIn("LD_LIBRARY_PATH=/opt/ct2-runtime/lib", text)
        with open(os.path.join(os.path.dirname(__file__),
                               "../bases/fasterwhisper/probe_ct2_cuda.py")) as fh:
            probe = fh.read()
        self.assertIn("libcublas.so.12", probe)
        self.assertIn("libcublas.so.13", probe)
        self.assertIn("libcudnn.so", probe)
        with open(os.path.join(os.path.dirname(__file__),
                               "../bases/fasterwhisper/collect_ct2_runtime.py")) as fh:
            collect = fh.read()
        self.assertIn("libcublas.so", collect)
        self.assertIn("nvidia_wheel_libs", collect)
        self.assertIn("system_cudnn_libs", collect)
        self.assertIn("libcudnn*.so*", collect)
        with open(os.path.join(os.path.dirname(__file__),
                               "../wrapper/caps/whisper.py")) as fh:
            whisper = fh.read()
        self.assertIn("_ct2tf.torch = torch", whisper)
        load = whisper.split("def _load():", 1)[1].split("\ndef ", 1)[0]
        self.assertLess(load.find("import torch"),
                        load.find("from faster_whisper import WhisperModel"))


class SlimPyannoteRecipeTest(unittest.TestCase):
    def test_four_x_cannot_upgrade_the_torch_already_installed(self):
        path = os.path.join(os.path.dirname(__file__),
                            "../bases/pyannote/deps.Dockerfile")
        with open(path) as fh:
            text = fh.read()
        self.assertIn('pyannote.audio>=4,<5', text)
        self.assertIn("/tmp/torch.pin", text)
        self.assertIn("-c /tmp/torch.pin", text)
        self.assertNotIn("pyannote.audio>=3.3.0,<4", text)
        torch = text.find("torch torchaudio --index-url")
        pin = text.find("pip freeze | grep -E '^(torch|torchaudio)=='")
        four = text.find('"pyannote.audio>=4,<5"')
        self.assertGreater(pin, torch)
        self.assertGreater(four, pin)
        self.assertNotIn("pip uninstall", text)


class SlimTtsOvRecipeTest(unittest.TestCase):
    def test_intel_tts_bases_have_openvino_and_no_cuda(self):
        root = os.path.join(os.path.dirname(__file__), "../bases")
        for name in ("breezeov", "fireredov"):
            path = os.path.join(root, name, "deps.Dockerfile")
            with open(path) as fh:
                text = fh.read()
            self.assertNotIn("nvidia/cuda", text, path)
            self.assertIn("openvino", text, path)
            self.assertIn("download.pytorch.org/whl/cpu", text, path)


class SlimBreezeRecipeTest(unittest.TestCase):
    def test_numpy_and_sox_are_present_before_qwen_tts(self):
        path = os.path.join(os.path.dirname(__file__),
                            "../bases/breeze/deps.Dockerfile")
        with open(path) as fh:
            text = fh.read()
        self.assertIn("sox", text)
        numpy = text.find("pip install --no-cache-dir numpy")
        qwen = text.find('"qwen-tts==0.1.1"')
        self.assertGreater(numpy, -1)
        self.assertGreater(qwen, numpy)


class SlimRuntimeRecipeTest(unittest.TestCase):
    def _runtime_dir(self):
        return os.path.join(os.path.dirname(__file__), "../bases/runtime")

    def test_torch_install_and_cuda_strip_share_one_run(self):
        root = os.path.join(os.path.dirname(__file__), "../bases")
        for name in ("pyannote", "firered", "breeze", "qwen"):
            path = os.path.join(root, name, "deps.Dockerfile")
            with open(path) as fh:
                text = fh.read()
            self.assertNotIn("python3-venv", text, path)
            run = text.split("RUN set -eux;", 1)[1]
            pip = run.find("torch torchaudio")
            strip = run.find("sh /tmp/strip_unused_cuda.sh;")
            self.assertGreater(pip, -1, path)
            self.assertGreater(strip, pip, path)
            if name == "qwen":
                # qwen-asr --no-deps: skip demo extras. numpy before the wheel.
                self.assertIn("--no-deps", run, path)
                self.assertGreater(run.find("qwen-asr"), run.find("numpy"), path)
                self.assertIn("uninstall -y gradio gradio-client flask sox", run, path)
                self.assertNotIn("sox;", text, path)

    def test_strip_script_drops_solvers_keeps_cublas_cudnn(self):
        with open(os.path.join(self._runtime_dir(), "strip_unused_cuda.sh")) as fh:
            text = fh.read()
        # Keep/drop follows ldd so a leftover CUDA 13 wheel cannot hide behind a shared soname.
        self.assertIn("resolved_files", text)
        self.assertIn("\ncd /\n", text)
        self.assertIn("libnccl.so", text)
        self.assertIn("keep-real", text)
        self.assertIn("libnvshmem_host.so", text)
        self.assertIn("nvidia-nvshmem", text)
        m = re.search(r"grep -iE '([^']+)'", text)
        self.assertIsNotNone(m)
        posix = m.group(1)
        py = posix.replace(r"[^[:space:]=]*", r"\S*")
        freeze = (
            "nvidia-cublas-cu12==12.8\n"
            "nvidia-cudnn-cu12==9.1\n"
            "nvidia-cublas==13.1\n"
            "nvidia-nccl-cu12==2.21\n"
            "nvidia-cusparselt-cu12==0.8\n"
            "cuda-toolkit==13.0.3\n"
            "triton==3.2.0\n"
        )
        candidates = [x.group(0).split("=")[0] for x in re.finditer(py, freeze, re.I | re.M)]
        self.assertEqual(candidates, [
            "nvidia-cublas-cu12",
            "nvidia-cudnn-cu12",
            "nvidia-cublas",
            "nvidia-nccl-cu12",
            "nvidia-cusparselt-cu12",
            "cuda-toolkit",
            "triton",
        ])


class BaseRegistrationTests(unittest.TestCase):
    """A base directory, the routing table and the README table name the same 12 bases.

    Adding a base means adding a key to catalog.BASES: app.py routes on AUDIO_BASE, which
    append-image.sh sets from the directory name, and a base missing from that table exits at
    startup saying the image was built wrong. Nothing here compared the two, so the way to find
    out was to build the image and run it. The README table carries the line "Keep this table in
    sync whenever a base is added", which until now was a sentence asking to be remembered.

    Both walk bases/ rather than a list written here, so a base added tomorrow is in scope
    without anyone editing this file. append.env is what makes a directory a base: every one has
    it and bases/runtime, which is shared build scripts rather than a base, does not. nemo is why
    the marker is not deps.Dockerfile -- it layers on another image and has no deps of its own.
    """

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _bases(self):
        found = {name for name in os.listdir(os.path.join(self.ROOT, "bases"))
                 if os.path.isfile(os.path.join(self.ROOT, "bases", name, "append.env"))}
        self.assertTrue(found, "no bases/*/append.env found; this test has lost its anchor")
        return found

    def test_every_base_directory_is_routable(self):
        self.assertEqual(self._bases(), set(catalog.BASES))

    def test_every_base_directory_has_a_readme_row(self):
        with open(os.path.join(self.ROOT, "README.md"), encoding="utf-8") as fh:
            rows = set(re.findall(r"^\| `([a-z0-9-]+)` \| `beclab/", fh.read(), re.M))
        self.assertEqual(self._bases(), rows)


class OnnxPinTests(unittest.TestCase):
    """The version the images install and the version CI tests against are one version.

    bases/speakrs-ov/deps.Dockerfile says onnx decides whether the derived model is accepted
    -- the derivation runs infer_shapes and the checker over a graph it edited -- and its
    comment states the image and the test run against the same release. Nothing held that
    claim up: the workflow installed onnx unpinned, so the two could differ for months with
    every run green. This is the cheapest thing that holds it, because it compares the two
    files as text and needs neither onnx installed nor a particular host.
    """

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _pins(self, path, pattern):
        with open(os.path.join(self.ROOT, path), encoding="utf-8") as fh:
            return re.findall(pattern, fh.read())

    def test_ci_installs_the_version_the_images_install(self):
        ci = self._pins(".github/workflows/build-image.yml", r"onnx==([0-9][^\s\"']*)")
        self.assertEqual(len(ci), 1, "expected exactly one onnx pin in build-image.yml")
        images = []
        for dockerfile in sorted(glob.glob(os.path.join(self.ROOT, "bases", "*",
                                                        "deps.Dockerfile"))):
            rel = os.path.relpath(dockerfile, self.ROOT)
            images += [(rel, v) for v in self._pins(rel, r"onnx==([0-9][^\s\"']*)")]
        self.assertTrue(images, "no base pins onnx; drop this test with the last one")
        for rel, version in images:
            self.assertEqual(version, ci[0],
                             "%s installs onnx %s, CI tests against %s" % (rel, version, ci[0]))


class RawBodyIsReachedOnlyThroughItsWrapperTest(unittest.TestCase):
    """`_x_raw` is a body someone pulled a universal obligation out of. Only `_x` may call it.

    🔴 The failure this exists for has no symptom. `align._align` owns one line -- put the
    GPU's all-time peak back when a model call raises -- and that line was written out by
    hand at four call sites and missed a site every time: the enumeration said three paths,
    then five, then seven, and the eighth was found by listing callers rather than by
    thinking harder. A missed site does not raise, does not log and does not change a
    response; it freezes the cost model, because `_used` answers zero for every call that
    stays below a peak nothing reset. **Extraction is what fixed it, and extraction only
    holds while the raw body stays unreachable.** One `res = _align_raw(...)` typed at a new
    call site -- by someone copying the line above it, which is the normal way a call site is
    written -- puts the bug straight back with nothing red.

    Deliberately a convention and not one hard-coded name: the next obligation somebody
    pulls out of a function gets this check by naming the body `_<name>_raw`, with no edit
    here. `wrapper/` only -- a test that stubs the raw body is reaching past the wrapper on
    purpose, which is the one place doing so is right.
    """

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @staticmethod
    def _offences(source, where):
        """[(where, line, raw_name, enclosing function)] for every reference out of place.

        Reads names AND attributes: `_align_raw(...)` inside the module and
        `mod._align_raw(...)` from another one are the same mistake, and only the first is a
        bare Name. The enclosing function is tracked by walking the tree rather than by
        `ast.walk`, because walk loses the nesting that decides whether a reference is legal.
        """
        import ast
        found = []

        def visit(node, fn):
            here = fn
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                here = node.name
            for child in ast.iter_child_nodes(node):
                name = (child.id if isinstance(child, ast.Name)
                        else child.attr if isinstance(child, ast.Attribute) else None)
                # A def of the raw body is not a reference to it.
                if (name and name.startswith("_") and name.endswith("_raw")
                        and here != name[:-len("_raw")]):
                    found.append((where, child.lineno, name, here))
                visit(child, here)

        visit(ast.parse(source), None)
        return found

    def test_nothing_in_the_wrapper_reaches_a_raw_body_except_its_wrapper(self):
        import ast
        bad, raw_bodies = [], []
        for path in sorted(glob.glob(os.path.join(self.ROOT, "wrapper", "**", "*.py"),
                                     recursive=True)):
            rel = os.path.relpath(path, self.ROOT)
            with open(path, encoding="utf-8") as fh:
                source = fh.read()
            raw_bodies += [n.name for n in ast.walk(ast.parse(source))
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                           and n.name.startswith("_") and n.name.endswith("_raw")]
            bad += self._offences(source, rel)
        # 🔴 A ratchet on the anchor, not on the offences. With no `_x_raw` defined anywhere
        # the loop above finds nothing and this file passes while saying nothing -- which is
        # what it would do the day someone renames `_align_raw` to `_align_inner` and keeps
        # every call site. Name the convention here so the rename has to come past it.
        self.assertIn("_align_raw", raw_bodies,
                      "wrapper/ defines no _align_raw, so this check is watching an empty "
                      "set. Either the extraction was undone -- then the peak-reset is back "
                      "to being written out by hand at every call site -- or the body was "
                      "renamed out of the _<name>_raw convention this check reads")
        self.assertEqual(bad, [], "\n".join([
            "a raw body is referenced from outside the wrapper that owns it.",
            "`_x_raw` exists because `_x` does something at EVERY call -- for _align_raw,",
            "putting the GPU peak back when the call raises. Reaching past `_x` skips it,",
            "silently: nothing raises, nothing logs, and the cost model stops correcting.",
            "Call `_x`.", *["%s:%d refers to %s from %s" % b for b in bad]]))

    def test_the_scan_sees_the_shapes_a_call_site_is_written_in(self):
        """🔴 A positive control. The assertion above is an empty-set assertion on a clean
        tree, so it passes just as well after the scan goes blind -- which is how a check
        like this normally dies: someone reformats, the pattern stops matching, and the
        green stays green. These are the shapes themselves.
        """
        cases = {
            "def f():\n    return _align_raw(1)\n": 1,               # bare name
            "def f():\n    return align._align_raw(1)\n": 1,         # through a module
            "def f():\n    g = _align_raw\n": 1,                     # taken, not called
            "def _align():\n    return _align_raw(1)\n": 0,          # its own wrapper
            "def _align():\n    def inner():\n        return _align_raw(1)\n": 1,
            "def _align_raw():\n    return 1\n": 0,                  # the definition
        }
        for source, expected in cases.items():
            self.assertEqual(
                len(self._offences(source, "<control>")), expected,
                "the raw-body scan reads %r as %d offence(s), not %d -- so every call site "
                "written that way is invisible to the assertion above"
                % (source, len(self._offences(source, "<control>")), expected))


if __name__ == "__main__":
    unittest.main()
