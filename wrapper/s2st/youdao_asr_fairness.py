"""Keep original ASR computation cooperative with WebSocket control traffic."""

import ast


def apply(code):
    tree = ast.parse(code)
    routes = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef)
              and node.name == "asr_stream_api_v1"]
    processors = [node for route in routes for node in route.body
                  if isinstance(node, ast.AsyncFunctionDef) and node.name == "processor"]
    if len(routes) != 1 or len(processors) != 1:
        raise ValueError("Unknown original ASR processor")
    loops = [node for node in ast.walk(processors[0]) if isinstance(node, ast.While)]
    if len(loops) != 2:
        raise ValueError("Unknown original ASR processor loops")
    lines = code.splitlines(keepends=True)
    for loop in sorted(loops, key=lambda node: node.body[0].lineno, reverse=True):
        first = loop.body[0]
        lines.insert(first.lineno - 1, " " * first.col_offset + "await asyncio.sleep(0)\n")
    result = "".join(lines)
    compile(result, "<cooperative-original-asr>", "exec")
    return result
