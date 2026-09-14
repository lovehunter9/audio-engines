#!/usr/bin/env python3
"""Patch a checked-out openvino.genai tree so Qwen3-ASR generate() takes many waveforms.

Public generate() only accepted one vector<float>. split_audio_into_chunks already
knows a list of waveforms and stamps orig_batch; merge_chunk_results already folds
chunks back to one text per input. This patch only opens that binding.

infer() stays the upstream serial encode+decode. A stacked decoder.generate() was
tried (intel-ov17) and failed on device: the compiled decoder takes one encoder
tensor, so every span came back as the first clip's text (or empty). Do not put
that path back without a per-clip encoder that the IR actually batches.
"""
from __future__ import annotations

import pathlib
import sys

AUDIO_INPUTS_OLD = "using AudioInputs = std::variant<std::vector<float>>;"
AUDIO_INPUTS_NEW = (
    "using AudioInputs = std::variant<std::vector<float>, std::vector<std::vector<float>>>;"
)

GENERATE_UNWRAP_OLD = """    const std::vector<float>& audio = std::visit(
        ov::genai::utils::overloaded{
            [](const std::vector<float>& input) -> const std::vector<float>& {
                return input;
            },
        },
        audio_inputs);

    const std::vector<AudioChunk> chunks =
        split_audio_into_chunks({audio}, m_feature_extractor.sampling_rate, MAX_ASR_INPUT_SECONDS);"""

GENERATE_UNWRAP_NEW = """    std::vector<std::vector<float>> audios;
    std::visit(ov::genai::utils::overloaded{
                   [&](const std::vector<float>& input) {
                       audios = {input};
                   },
                   [&](const std::vector<std::vector<float>>& input) {
                       audios = input;
                   },
               },
               audio_inputs);

    const std::vector<AudioChunk> chunks =
        split_audio_into_chunks(audios, m_feature_extractor.sampling_rate, MAX_ASR_INPUT_SECONDS);"""

GENERATE_UNWRAP_OLD_LOOSE = "split_audio_into_chunks({audio},"
GENERATE_UNWRAP_NEW_LOOSE = "split_audio_into_chunks(audios,"


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: pathlib.Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print("patched", path)


def patch_audio_inputs(root: pathlib.Path) -> None:
    hits = [
        p for p in root.rglob("pipeline.hpp")
        if "automatic_speech_recognition" in str(p) and "using AudioInputs" in p.read_text(encoding="utf-8")
    ]
    if not hits:
        raise SystemExit("pipeline.hpp with AudioInputs not found")
    path = hits[0]
    text = _read(path)
    if AUDIO_INPUTS_NEW in text:
        print("AudioInputs already patched", path)
        return
    if AUDIO_INPUTS_OLD in text:
        _write(path, text.replace(AUDIO_INPUTS_OLD, AUDIO_INPUTS_NEW, 1))
        return
    loose = "using AudioInputs = std::variant<"
    i = text.find(loose)
    if i < 0:
        raise SystemExit("AudioInputs typedef not found in %s" % path)
    j = text.find(";", i)
    _write(path, text[:i] + AUDIO_INPUTS_NEW + text[j + 1 :])


def patch_generate_unwrap(src: str, path: str) -> str:
    if GENERATE_UNWRAP_OLD in src:
        return src.replace(GENERATE_UNWRAP_OLD, GENERATE_UNWRAP_NEW, 1)
    if "split_audio_into_chunks({audio}," not in src:
        if "split_audio_into_chunks(audios," in src:
            print("generate unwrap already patched", path)
            return src
        raise SystemExit("split_audio_into_chunks({audio} not found in %s" % path)
    src = src.replace(
        "const std::vector<float>& audio = std::visit(",
        "std::vector<std::vector<float>> audios;\n    std::visit(",
        1,
    )
    src = src.replace(GENERATE_UNWRAP_OLD_LOOSE, GENERATE_UNWRAP_NEW_LOOSE, 1)
    if "[&](const std::vector<std::vector<float>>& input)" not in src:
        src = src.replace(
            "[](const std::vector<float>& input) -> const std::vector<float>& {\n                return input;\n            },",
            "[&](const std::vector<float>& input) { audios = {input}; },\n"
            "                   [&](const std::vector<std::vector<float>>& input) { audios = input; },",
            1,
        )
    return src


OTHER_VISIT_ARM = """
            [](const std::vector<std::vector<float>>&) -> const std::vector<float>& {
                OPENVINO_THROW("batched audio is only implemented for Qwen3-ASR");
                static const std::vector<float> empty;
                return empty;
            },"""


def patch_other_backends(root: pathlib.Path) -> None:
    """std::visit must be exhaustive after AudioInputs gains a second alternative."""
    needle = "[](const std::vector<float>& input) -> const std::vector<float>& {"
    for path in list(root.rglob("*.cpp")) + list(root.rglob("*.hpp")):
        if "qwen3-asr" in str(path):
            continue
        src = _read(path)
        if needle not in src or "batched audio is only implemented" in src:
            continue
        i = src.find(needle)
        brace = src.find("},", i)
        if brace < 0:
            raise SystemExit("could not close AudioInputs visit in %s" % path)
        _write(path, src[: brace + 2] + OTHER_VISIT_ARM + src[brace + 2 :])


def patch_qwen3_pipeline(root: pathlib.Path) -> None:
    hits = list(root.rglob("qwen3-asr/pipeline.cpp"))
    if not hits:
        raise SystemExit("qwen3-asr/pipeline.cpp not found")
    path = hits[0]
    src = _read(path)
    if "stack_encoder_hiddens" in src:
        raise SystemExit(
            "%s still has stack_encoder_hiddens; that path copies the first "
            "clip onto the whole group and must not ship" % path
        )
    src = patch_generate_unwrap(src, str(path))
    _write(path, src)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_qwen3_asr_batch.py <openvino.genai checkout>")
    root = pathlib.Path(sys.argv[1]).resolve()
    if not root.is_dir():
        raise SystemExit("not a directory: %s" % root)
    patch_audio_inputs(root)
    patch_qwen3_pipeline(root)
    patch_other_backends(root)
    print("qwen3-asr multi-audio patch applied")


if __name__ == "__main__":
    main()
