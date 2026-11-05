import ast
import pathlib
import unittest

from wrapper import catalog


KNOBS = {"speed", "stability", "similarity_boost", "style", "use_speaker_boost"}


class MingVoiceSettingsCatalogTest(unittest.TestCase):
    def test_cuda_and_openvino_advertise_all_five_el_controls(self):
        for base in ("ming", "mingov"):
            with self.subTest(base=base):
                self.assertEqual(catalog.module_of(base, "tts"), "ming")
                caps = catalog.implements(base)
                rows = catalog.spec_endpoints(base, caps, caps)
                for operation in ("speech.synthesize", "speech.synthesize.stream", "voice.settings.update"):
                    with self.subTest(base=base, operation=operation):
                        row = next(row for row in rows if row["protocol"] == "elevenlabs.voice.v1" and row["operation_id"] == operation)
                        names = [parameter["name"] for parameter in row["parameters"]]
                        self.assertTrue(KNOBS.issubset(names))
                        self.assertEqual(len(names), len(set(names)))

    def test_five_control_descriptors_match_existing_engines(self):
        for module in ("ming", "qwen3_voice", "breeze", "firered"):
            for path in ("/v1/text-to-speech/{voice_id}", "/v1/text-to-speech/{voice_id}/stream", "/v1/voices/{voice_id}/settings/edit"):
                method, operation = ("POST", "voice.settings.update") if path.endswith("/edit") else ("POST", "speech.synthesize.stream" if path.endswith("/stream") else "speech.synthesize")
                descriptor = catalog.describe_endpoint(module, "tts", method, path, False)
                controls = {parameter["name"]: parameter for parameter in descriptor["parameters"] if parameter["name"] in KNOBS}
                reference = {parameter["name"]: parameter for parameter in catalog.describe_endpoint("qwen3_voice", "tts", method, path, False)["parameters"] if parameter["name"] in KNOBS}
                with self.subTest(module=module, operation=operation):
                    self.assertEqual(controls, reference)

    def test_declared_defaults_and_ranges_match_existing_el_runtime(self):
        source = pathlib.Path(__file__).parents[1] / "wrapper/caps/tts_el.py"
        tree = ast.parse(source.read_text())
        assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_DEFAULT_SETTINGS" for target in node.targets))
        defaults = ast.literal_eval(assignment.value)
        params = catalog.describe_endpoint("ming", "tts", "POST", "/v1/text-to-speech/{voice_id}", True)["parameters"]
        controls = {parameter["name"]: parameter for parameter in params if parameter["name"] in KNOBS}
        self.assertEqual({name: parameter["default"] for name, parameter in controls.items()}, defaults)
        self.assertEqual((controls["speed"]["minimum"], controls["speed"]["maximum"]), (0.25, 4.0))
        for name in ("stability", "similarity_boost", "style"):
            self.assertEqual((controls[name]["minimum"], controls[name]["maximum"]), (0.0, 1.0))
