import ast
from pathlib import Path

import pytest

source = Path(__file__).parents[1] / "wrapper/caps/seamless_native.py"
node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "_segments")
namespace = {}
exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
segments = namespace["_segments"]


@pytest.mark.parametrize("length", [400, 401, 5119, 5120, 5121, 5280, 5519, 5520, 10240, 168960, 585920])
@pytest.mark.parametrize("packet", [160, 5120, 6337])
def test_final_nonempty_frame_preserves_every_sample(length, packet):
    pending, sent = [], []
    for start in range(0, length, packet):
        pending.extend(range(start, min(start + packet, length)))
        sent.extend(segments(pending, False))
    sent.extend(segments(pending, True))
    assert not pending
    assert [sample for chunk, _ in sent for sample in chunk] == list(range(length))
    assert [last for _, last in sent] == [False] * (len(sent) - 1) + [True]
    assert len(sent[-1][0]) >= 400


def test_short_input_is_rejected_without_replaying_previous_audio():
    with pytest.raises(ValueError, match="25 milliseconds"):
        list(segments([0] * 399, True))
    assert list(segments([], True)) == []
