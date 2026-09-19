"""Sizing one call by what it costs. ⚠️ Nothing here is measured: the padding rule holds for
transformers v4.57.6, the one constant is arithmetic off the model's own config, and
`bases/qwen/deps.Dockerfile` leaves qwen-asr itself unpinned."""
import math


#: What a span costs a call over and above its audio, as the audio seconds that cost the same.
#: 🔴 Derived, not measured, the way the host floor is. Two things in a call scale with the SPAN
#: COUNT and not with the audio: the logits, `[N, vocab]` at 151,936 x 4 bytes = 0.58 MiB a span,
#: and the KV for the prompt that precedes every span's audio, 2 x 28 layers x 8 kv-heads x 128
#: head-dim x 2 bytes = 112 KiB a token. At 16 to 40 prompt tokens that is 2.3 to 5.0 MiB a span,
#: or 0.39 to 0.83 seconds at the opening figure of 6 MiB a padded second. Rounded UP to one
#: second, because the derivation counts tensors and not the workspace around them.
#:
#: ⚠️ Read off `Qwen/Qwen3-ASR-1.7B`'s config, so it is a property of that checkpoint. It is a
#: FLOOR: every term left out is on the conservative side. Where spans are long it is noise --
#: 2.3% at 30 s -- and where they are short it is the whole story: 35% at 2 s, and the diarizer
#: upstream has no minimum segment length at all.
SPAN_FLOOR_SECONDS = 1.0


# 🔴 The unit is padded audio seconds, not spans: `audio_kwargs.padding` is True with truncation
# off, so `WhisperFeatureExtractor` pads every clip in a group up to the longest one.
def padded_seconds(seconds_list):
    """What a group costs: member count times the longest member, not `sum` -- the same 32 spans
    cost four times as much when one of them is four times longer -- plus the floor every span
    carries whatever its length. 🔴 The floor is why a count can bound memory at all: without it
    a call of 512 two-second spans priced the same as 32 of them carrying the same audio."""
    if not seconds_list:
        return 0.0
    return len(seconds_list) * (max(seconds_list) + SPAN_FLOOR_SECONDS)


def group_capacity(longest_seconds, budget_padded_seconds, max_spans=None):
    """How many clips of at most `longest_seconds` fit one call. Never less than one -- one, not
    zero, for a clip over the whole budget: the engine tries it and splits on failure."""
    if budget_padded_seconds > 0 and longest_seconds > 0:
        fits = int(budget_padded_seconds // (longest_seconds + SPAN_FLOOR_SECONDS))
    else:
        fits = 1
    # `is not None`, not truthiness: a count of zero is a bound somebody wrote, and reading
    # it as "no bound" is the direction that makes a call bigger.
    if max_spans is not None:
        fits = min(fits, int(max_spans))
    return max(1, fits)


# 🔴 Longest first is load-bearing: sorted descending the first member of a group IS its longest,
# so `group_capacity` is exact and no 300 s span is padded onto a call full of 5 s ones.
def pack(spans, budget_padded_seconds, max_spans=None):
    """Groups of one call each, longest first: `spans` is [(index, seconds)] -> [[index]]. Every
    group carries the caller's index, so the caller's own order survives in its results.

    Fewest calls; among those, the smallest single call; among those, the least work overall.

    🔴 The middle one is the memory objective, and it is the PEAK rather than the total because
    calls run one at a time -- the card only ever holds one group, so the total is time and the
    largest group is what fails. Choosing the peak costs a median 5.7% more total work and takes
    a median 17.8% off the peak, at no cost in calls at all.

    🔴 Calls come first because nothing else can: any grouping of unequal spans pads, so putting
    work first would mean one span a call and no batching. A call also has a fixed cost that does
    not shrink with the audio in it -- halving the calls over one measured request took 28% off
    the clock. ⚠️ That is two points on one card; what it supports is the ordering, not a number,
    and neither number is in this file.

    ⚠️ Where they genuinely trade, calls win: a 300 s and a 120 s span go together at 600
    padded seconds rather than separately at 420, because that really does save a call.

    🔴 Where they do not trade, filling was pure loss. One 300 s span and twenty 5 s ones under
    a 600 s budget came out `[300, 5] + [5 x 19]`, spending 695 padded seconds -- 295 of them to
    carry five seconds of audio -- for the same two calls that `[300] + [5 x 20]` does in 400.
    That span had a free seat in the next group all along.
    """
    order = sorted(spans, key=lambda s: float(s[1]), reverse=True)
    n = len(order)
    if not n:
        return []
    # The best plan for the spans from i onwards, read back through take[i]. 🔴 Searched rather
    # than a rule of thumb: the alternative is a slack threshold nobody has measured, and the
    # thing this replaces was a rule of thumb that looked obviously right.
    calls = [0] * (n + 1)
    peak = [0.0] * (n + 1)
    cost = [0.0] * (n + 1)
    take = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        head = float(order[i][1]) + SPAN_FLOOR_SECONDS
        most = group_capacity(float(order[i][1]), budget_padded_seconds, max_spans)
        if most >= n - i:
            # Everything left fits one call, so one call is the fewest there can be, and with one
            # group the peak and the total are the same number. Without this the loop below walks
            # every size on a request of short spans, the shape where capacity is largest: 2000
            # of them cost 100 ms.
            calls[i], peak[i], cost[i], take[i] = 1, (n - i) * head, (n - i) * head, n - i
            continue
        best = None
        for size in range(1, min(most, n - i) + 1):
            here = size * head
            key = (1 + calls[i + size], max(here, peak[i + size]), here + cost[i + size])
            if best is None or key < best[0]:
                best = (key, size)
        (calls[i], peak[i], cost[i]), take[i] = best[0], best[1]
    groups, at = [], 0
    while at < n:
        size = take[at]
        groups.append([i for i, _s in order[at:at + size]])
        at += size
    return groups


def budget_from_bytes(headroom_bytes, bytes_a_padded_second, fraction=1.0):
    """Padded seconds that fit `headroom_bytes`. 🔴 None when either input is unusable, 0.0 when
    the headroom was read and is spent: one value for both grows the batch when memory is least."""
    try:
        headroom = float(headroom_bytes)
        per_second = float(bytes_a_padded_second)
    except (TypeError, ValueError):
        # None included: it is what every reader in this module returns for "could not be
        # read", and it arrives here as a TypeError like any other unusable value.
        return None
    if headroom <= 0:
        # Read, and spent. A grant over its limit is the same answer as one exactly at it.
        return 0.0
    if per_second <= 0:
        return None
    value = fraction * headroom / per_second
    if not math.isfinite(value) or value < 0:
        return None
    return value
