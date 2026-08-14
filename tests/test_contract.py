import importlib
import json
import os
import sys
import unittest
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from wrapper import catalog, contract, gpu, tasks

EXPECTED_CAPABILITY_ENDPOINTS = {
    ("align", "align", "POST", "/v1/audio/align"): {"async_supported": True},
    # No /v1/audio/voices: the audio.cpp TTS families have no built-in speakers.
    ("audiocpp_tts", "tts", "POST", "/v1/audio/speech"): {"async_supported": True},
    ("audiocpp_tts", "tts", "POST", "/v1/audio/speech/batch"): {
        "async_supported": True,
    },
    ("audiocpp_tts", "tts", "WS", "/v1/audio/speech/stream"): {
        "async_supported": False,
    },
    # Shared /speech lives under tts; tts_clone only advertises the multipart clone route.
    ("audiocpp_tts", "tts_clone", "POST", "/v1/audio/speech/clone"): {
        "async_supported": True,
    },
    # No WS: ggml synthesizes whole utterances, so stream=1 is sentence-scoped.
    ("crispasr_tts", "tts", "GET", "/v1/audio/voices"): {"async_supported": False},
    ("crispasr_tts", "tts", "POST", "/v1/audio/speech"): {"async_supported": True},
    ("crispasr_tts", "tts", "POST", "/v1/audio/speech/batch"): {
        "async_supported": True,
    },
    ("diar", "diar", "POST", "/v1/audio/diarization"): {"async_supported": True},
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

TASK_PATHS = {
    "/v1/tasks",
    "/v1/tasks/{id}",
    "/v1/tasks/{id}/result",
}


class CatalogContractTest(unittest.TestCase):
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

    def test_audiocpp_serving_both_caps_does_not_repeat_a_path(self):
        # These models declare tts and tts_clone together. A path listed under both caps is what
        # the dashboard then prints twice, so the catalog must not share speech routes that way.
        rows = catalog.spec_endpoints("audiocpp", ["tts", "tts_clone"], ["tts", "tts_clone"])
        paths = [(r["method"], r["path"]) for r in rows if r["available"]]
        self.assertEqual(len(paths), len(set(paths)), paths)

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
            engine = Runtime("default-name", default_repo="default/repo", model=None)

        self.assertEqual(engine.model_name, "served-name")
        self.assertEqual(engine.model_repo, "org/repo")
        self.assertEqual(engine.port, 9123)
        self.assertEqual(engine.log_level, "warning")
        self.assertEqual(engine.state, {"ready": False, "error": None, "model": None})
        engine.state.update(ready=True)
        self.assertTrue(engine.state["ready"])

    def test_runtime_preserves_background_and_blocking_startup_axes(self):
        from wrapper.runtime import Runtime

        events = []
        background = Runtime("background")
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
            load = lambda: events.append("load")

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
        blocking = Runtime("blocking")
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
                "model": "Qwen/Qwen3-ForcedAligner-0.6B",
                "repo": "Qwen/Qwen3-ForcedAligner-0.6B",
                "state": {"ready": False, "error": None, "model": None, "device": "cpu"},
            },
            "diar": {
                "supports": ["diar"],
                "watchdog": "pyannote pipeline",
                "model": "pyannote-community-1",
                "repo": "pyannote-community-1",
                "state": {
                    "ready": False,
                    "error": None,
                    "pipeline": None,
                    "device": "cpu",
                    "batch1": False,
                },
            },
            "diar_stream": {
                "supports": ["diar_stream"],
                "watchdog": "streaming sortformer",
                "model": "diar-streaming-sortformer",
                "repo": "nvidia/diar_streaming_sortformer_4spk-v2.1",
                "state": {"ready": False, "error": None, "model": None, "device": "cpu"},
                "disable_ws_ping": True,
            },
            "embed": {
                "supports": ["speaker_embed"],
                "watchdog": "pyannote embedding",
                "model": "pyannote-embedding",
                "repo": "pyannote-embedding",
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
                "model": "mtl-mimic-voicebank",
                "repo": "mtl-mimic-voicebank",
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
                "model": "Qwen/Qwen3-ASR-1.7B",
                "repo": "Qwen/Qwen3-ASR-1.7B",
                "state": {"ready": False, "error": None, "asr": None},
                "load_on_main": True,
                "disable_ws_ping": True,
            },
            "vad": {
                "supports": ["vad"],
                "watchdog": "silero-vad",
                "model": "silero-v5",
                "repo": "silero-v5",
                "state": {"ready": False, "error": None, "model": None, "get_ts": None},
            },
            "whisper": {
                "supports": ["stt"],
                "watchdog": "faster-whisper",
                "model": "Systran/faster-whisper-large-v3",
                "repo": "Systran/faster-whisper-large-v3",
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
                        "MODEL_NAME": expected["model"],
                        "MODEL_SOURCE": "",
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
                    self.assertEqual(module.MODEL_NAME, expected["model"])
                    self.assertEqual(module._runtime.model_repo, expected["repo"])
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
        from fastapi import HTTPException

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

        self.assertEqual(spec["schema_version"], 1)
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


if __name__ == "__main__":
    unittest.main()
