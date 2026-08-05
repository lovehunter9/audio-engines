import json

from fastapi import HTTPException


def parse_segments(raw):
    try:
        segments = json.loads(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid `segments` json: {e}")
    if not isinstance(segments, list):
        raise HTTPException(status_code=400, detail="`segments` must be a JSON array")
    return segments
