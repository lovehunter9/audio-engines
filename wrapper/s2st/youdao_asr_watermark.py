import ast
import re


def apply(code):
    tree = ast.parse(code)
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "asr_stream_api_v1"
    )
    lines = code.splitlines(True)
    index = function.body[0].lineno - 1
    lines.insert(index, "    _s2st_wire_samples = 0\n    _s2st_eos_received = False\n")
    code = "".join(lines)
    before, handler = code.split("async def asr_stream_api_v1(", 1)
    needle = "            if use_vad:\n                stream_vad.reset()"
    if handler.count(needle) != 1:
        raise RuntimeError("Unexpected VAD request initialization")
    handler = handler.replace(needle, "            if True:\n                stream_vad.reset()", 1)
    needle = "                audio_buf = np.concatenate((audio_buf, audio_seg))"
    if handler.count(needle) != 1:
        raise RuntimeError("Unexpected ASR PCM append boundary")
    handler = handler.replace(
        "        nonlocal audio_buf, tmp_audio_pointer, is_first_seg, total_secs",
        "        nonlocal _s2st_wire_samples, _s2st_eos_received\n        nonlocal audio_buf, tmp_audio_pointer, is_first_seg, total_secs",
        1,
    )
    handler = handler.replace(
        needle, "                _s2st_wire_samples += len(audio_seg)\n" + needle, 1
    )
    pattern = r"(?m)^([ \t]*)if data == YOUDAO_ONETIME_ASR_EOS_STRING:[ \t]*$"
    matches = list(re.finditer(pattern, handler))
    if len(matches) != 1:
        raise RuntimeError("Unexpected ASR EOS marker")
    match = matches[0]
    handler = (
        handler[: match.end()]
        + "\n"
        + match[1]
        + "    _s2st_eos_received = True"
        + handler[match.end() :]
    )
    needle = 'out_str = {"status": "success", "requestId": f"{requestId}", "msg": out_msg}'
    pattern = r"(?m)^( +)" + re.escape(needle)
    count = 0

    def replace(match):
        nonlocal count
        count += 1
        indent = match[1]
        return (
            indent
            + 'out_msg["source_samples"] = (_s2st_wire_samples if _s2st_eos_received else _s2st_wire_samples - len(audio_buf) + tmp_audio_pointer)\n'
            + indent
            + needle
        )

    handler = re.sub(pattern, replace, handler)
    if count != 4:
        raise RuntimeError("Unexpected ASR response boundaries: " + str(count))
    return before + "async def asr_stream_api_v1(" + handler
