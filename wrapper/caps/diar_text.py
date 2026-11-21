# MOSS-Transcribe-Diarize's transcript format, shared by the transformers and SGLang-Omni caps.
import re

MOSS_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)

_MOSS_SEGMENT = re.compile(r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]", re.S)


def parse_moss(text):
    """Segments of MOSS's `[start][Sxx]text[end]` transcript; anything unparseable is dropped."""
    out = []
    for m in _MOSS_SEGMENT.finditer(text or ""):
        words = m.group(3).strip()
        if words:
            out.append({"speaker": m.group(2), "start": float(m.group(1)),
                        "end": float(m.group(4)), "text": words})
    return out


def joined_text(segments):
    """The words of the segments in time order, without speakers."""
    parts = []
    for s in sorted(segments, key=lambda s: s["start"]):
        if parts and not _cjk(parts[-1][-1]) and not _cjk(s["text"][0]):
            parts.append(" ")
        parts.append(s["text"])
    return "".join(parts)


def _cjk(ch):
    return "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af"


def turns(segments, exclusive):
    """Speaker turns of the segments, in time order.

    Exclusive turns hold one speaker at a time: where two overlap, the one that started later
    takes over from its start, and touching turns of one speaker merge.
    """
    segs = sorted(({"start": round(float(s["start"]), 3), "end": round(float(s["end"]), 3),
                    "speaker": s["speaker"]} for s in segments), key=lambda s: (s["start"], s["end"]))
    segs = [s for s in segs if s["end"] > s["start"]]
    if not exclusive:
        return segs
    points = sorted({p for s in segs for p in (s["start"], s["end"])})
    out, active, i = [], [], 0
    for a, b in zip(points, points[1:]):
        while i < len(segs) and segs[i]["start"] <= a:
            active.append(segs[i])
            i += 1
        active = [s for s in active if s["end"] > a]
        if not active:
            continue
        speaker = active[-1]["speaker"]
        if out and out[-1]["speaker"] == speaker and out[-1]["end"] == a:
            out[-1]["end"] = b
        else:
            out.append({"start": a, "end": b, "speaker": speaker})
    return out
