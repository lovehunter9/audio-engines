"""Choose a fixed, audited ASR increment per source language, never per content."""
import ast
import re


def apply(code, default_ms, chinese_ms):
    allowed = (800, 1600, 2000, 4000)
    if default_ms not in allowed or chinese_ms not in allowed:
        raise ValueError("Unsupported ASR delivery increment")
    functions = [n for n in ast.parse(code).body if isinstance(n, ast.AsyncFunctionDef) and n.name == "asr_stream_api_v1"]
    if len(functions) != 1:
        raise ValueError("Unknown official ASR request entry")
    function = functions[0]
    lines = code.splitlines(keepends=True)
    body = "".join(lines[function.lineno-1:function.end_lineno])
    needle = '            language = header.get("language", "zhen")\n'
    if body.count(needle) != 1:
        raise ValueError("Unknown official ASR source-language parser")
    names = {"CHUNK_ASR_SECONDS": "_s2st_chunk_seconds", "CHUNK_ASR_SIZE": "_s2st_chunk_size", "STEP_MS": "_s2st_step_ms"}
    for old, new in names.items():
        if old not in body:
            raise ValueError("Missing official ASR increment use: " + old)
        body = re.sub(r"\b"+old+r"\b", new, body)
    body = body.replace(needle, needle + f'''            _s2st_step_ms = {chinese_ms} if language == "Chinese" else {default_ms}
            _s2st_chunk_seconds = _s2st_step_ms / 1000
            _s2st_chunk_size = round(_s2st_chunk_seconds * SAMPLING_RATE)
''', 1)
    lines[function.lineno-1:function.end_lineno] = [body]
    result = "".join(lines)
    compile(result, "<source-language-asr-increment>", "exec")
    return result
