#!/usr/bin/env python3
"""Patch a checked-out openvino.genai tree so Qwen3-ASR generate() takes many waveforms.

Decoder already runs batch_size>1. Public generate() only accepted one vector<float>
and infer() called encode+decode once per chunk. This opens the binding and sends
the group through one decoder.generate() (encoder stays per-clip; lengths are padded).
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

# Fallback if AudioChunk is not spelled that way in this tree.
GENERATE_UNWRAP_OLD_LOOSE = "split_audio_into_chunks({audio},"
GENERATE_UNWRAP_NEW_LOOSE = "split_audio_into_chunks(audios,"

INFER_LOOP_MARK = "    for (size_t batch = 0; batch < batch_size; ++batch) {"

INFER_BATCH_HELPER = r'''
ov::Tensor stack_encoder_hiddens(const std::vector<ov::Tensor>& hiddens) {
    OPENVINO_ASSERT(!hiddens.empty(), "stack_encoder_hiddens: empty");
    const size_t n = hiddens.size();
    const size_t hidden_dim = hiddens[0].get_shape().at(2);
    size_t max_t = 0;
    for (const auto& h : hiddens) {
        OPENVINO_ASSERT(h.get_shape().size() == 3 && h.get_shape()[0] == 1,
                        "encoder hidden states must be [1, T, H]");
        max_t = std::max(max_t, h.get_shape()[1]);
    }
    ov::Tensor out(ov::element::f32, {n, max_t, hidden_dim});
    float* dst = out.data<float>();
    std::fill_n(dst, n * max_t * hidden_dim, 0.f);
    for (size_t i = 0; i < n; ++i) {
        const size_t t = hiddens[i].get_shape()[1];
        std::memcpy(dst + i * max_t * hidden_dim,
                    hiddens[i].data<float>(),
                    t * hidden_dim * sizeof(float));
    }
    return out;
}

'''


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: pathlib.Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print("patched", path)


def _must_replace(text: str, old: str, new: str, where: str) -> str:
    if old not in text:
        raise SystemExit("marker not found in %s:\n%s" % (where, old[:180]))
    return text.replace(old, new, 1)


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
    # Trees sometimes wrap the template args with spaces.
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
    # Keep the visit as-is if the exact block drifted; still pass a list of audios.
    src = src.replace(
        "const std::vector<float>& audio = std::visit(",
        "std::vector<std::vector<float>> audios;\n    std::visit(",
        1,
    )
    # After visit, `audio` no longer exists; rewrite the one-element wrap.
    src = src.replace(GENERATE_UNWRAP_OLD_LOOSE, GENERATE_UNWRAP_NEW_LOOSE, 1)
    if "[&](const std::vector<std::vector<float>>& input)" not in src:
        src = src.replace(
            "[](const std::vector<float>& input) -> const std::vector<float>& {\n                return input;\n            },",
            "[&](const std::vector<float>& input) { audios = {input}; },\n"
            "                   [&](const std::vector<std::vector<float>>& input) { audios = input; },",
            1,
        )
    return src


def patch_infer(src: str, path: str) -> str:
    if "stack_encoder_hiddens" in src:
        print("infer already patched", path)
        return src
    if INFER_LOOP_MARK not in src:
        raise SystemExit("infer loop not found in %s" % path)
    start = src.find(INFER_LOOP_MARK)
    end = src.find("    return results;", start)
    if end < 0:
        raise SystemExit("return results not found after infer loop in %s" % path)
    replacement = """    const bool force_serial = bool(streamer_ptr) || batch_size == 1;
    if (force_serial) {
""" + src[start:end] + """        return results;
    }

    std::vector<ov::Tensor> hiddens;
    std::vector<size_t> audio_token_counts;
    hiddens.reserve(batch_size);
    audio_token_counts.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        const auto encoder_start_time = std::chrono::steady_clock::now();
        ov::Tensor hidden = m_encoder->encode(features[i]);
        const auto encoder_stop_time = std::chrono::steady_clock::now();
        const auto encoder_infer_ms = PerfMetrics::get_microsec(encoder_stop_time - encoder_start_time);
        perf_metrics.raw_metrics.m_inference_durations[0] += MicroSeconds(encoder_infer_ms);
        perf_metrics.asr_raw_metrics.encode_inference_durations.emplace_back(encoder_infer_ms);
        audio_token_counts.push_back(hidden.get_shape()[1]);
        hiddens.push_back(std::move(hidden));
    }

    const std::vector<std::string> processed_prompts = extend_audio_tokens(prompts, audio_token_counts);
    const auto tokenization_start_time = std::chrono::steady_clock::now();
    const ov::Tensor input_ids = m_tokenizer.encode(processed_prompts).input_ids;
    const auto tokenization_stop_time = std::chrono::steady_clock::now();
    perf_metrics.raw_metrics.tokenization_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(tokenization_stop_time - tokenization_start_time)));

    const ov::Tensor encoder_batch = stack_encoder_hiddens(hiddens);
    const auto encoded_results = m_decoder->generate(input_ids,
                                                     encoder_batch,
                                                     config,
                                                     perf_metrics.raw_metrics,
                                                     perf_metrics.asr_raw_metrics,
                                                     nullptr);

    const auto detokenization_start_time = std::chrono::steady_clock::now();
    for (size_t i = 0; i < batch_size; ++i) {
        results.push_back(m_tokenizer.decode(encoded_results.tokens[i]));
    }
    const auto detokenization_stop_time = std::chrono::steady_clock::now();
    perf_metrics.raw_metrics.detokenization_durations.emplace_back(
        MicroSeconds(PerfMetrics::get_microsec(detokenization_stop_time - detokenization_start_time)));
"""
    src = src[:start] + replacement + src[end:]
    if "#include <cstring>" not in src:
        src = src.replace("#include <algorithm>", "#include <algorithm>\n#include <cstring>", 1)
    if "#include <algorithm>" not in src:
        src = src.replace("#include \"pipeline.hpp\"", "#include \"pipeline.hpp\"\n#include <algorithm>", 1)
    ns = src.find("namespace ov::genai {")
    if ns < 0:
        raise SystemExit("namespace ov::genai not found in %s" % path)
    src = src[:ns] + INFER_BATCH_HELPER + src[ns:]
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
    src = patch_generate_unwrap(src, str(path))
    src = patch_infer(src, str(path))
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
