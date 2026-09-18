"""What the align path must do when the card refuses a call."""
import sys
import unittest

from wrapper.caps import align


class _Ctx:
    """The slice of a task context this path touches."""

    def __init__(self):
        self.metered = []

    def checkpoint(self):
        pass

    def meter(self, input_seconds=0):
        self.metered.append(input_seconds)

    def progress(self, **kw):
        pass


def _spans(n, seconds=30.0, positions=400):
    return [align._Span(i, None, "text %d" % i, "en", seconds, positions) for i in range(n)]


OOM = "CUDA out of memory. Tried to allocate 512.00 MiB"


class _Refusal(RuntimeError):
    """A refusal that can be weak-referenced."""


class AlignOomTest(unittest.TestCase):
    def setUp(self):
        self._call = align._call
        del align._groups[:]
        align._oom_count[0] = 0
        align._billed.clear()
        self.addCleanup(self._restore)

    def _restore(self):
        align._call = self._call
        del align._groups[:]
        align._oom_count[0] = 0
        align._billed.clear()

    def _run(self, fake_call, n=8):
        align._call = fake_call
        group = _spans(n)
        out = [None] * n
        ctx = _Ctx()
        return ctx, out, align._run_group(ctx, group, 16000, out, "test")

    def test_nothing_from_a_failed_call_is_still_alive_when_the_retry_starts(self):
        """🔴 The one that actually stands for the memory."""
        import gc
        import weakref

        raised = []
        survivors = []

        def fake_call(ctx, group, sr, out):
            gc.collect()
            survivors.append(sum(1 for r in raised if r() is not None))
            if len(group) > 1:
                e = _Refusal(OOM)
                raised.append(weakref.ref(e))
                raise e
            out[group[0].index] = {"language": "en", "units": []}
            align._bill(ctx, group[0])

        self._run(fake_call)
        self.assertGreater(len(raised), 0, "nothing was refused, so nothing was retried")
        self.assertEqual(survivors[1:], [0] * (len(survivors) - 1),
                         "a call that already failed was still reachable when the next "
                         "attempt began, so its activations were still on the card: %r"
                         % (survivors,))

    def test_no_exception_is_still_being_handled_when_a_retry_starts(self):
        """Revert `_attempt` to retrying inside the handler and this is what goes red.

        Measured on hzydemo01: nesting the handlers held each dead call's tensors alive,
        3.4 GiB of a 6 GiB grant on the first failure, and 96 spans became 191 calls that
        all failed.
        """
        handling = []

        def fake_call(ctx, group, sr, out):
            handling.append(sys.exc_info()[0])
            if len(group) > 1:
                raise RuntimeError(OOM)
            out[group[0].index] = {"language": "en", "units": []}
            align._bill(ctx, group[0])

        self._run(fake_call)
        self.assertGreater(len(handling), 1, "the group never split, so nothing was retried")
        self.assertEqual([h for h in handling if h is not None], [],
                         "a retry began while an earlier failure was still being handled, "
                         "which is what holds that call's memory")

    def test_a_refused_group_splits_until_it_fits_and_every_span_gets_an_answer(self):
        def fake_call(ctx, group, sr, out):
            if len(group) > 2:
                raise RuntimeError(OOM)
            for m in group:
                out[m.index] = {"language": "en", "units": []}
                align._bill(ctx, m)

        ctx, out, done = self._run(fake_call)
        self.assertEqual(done, 8)
        self.assertEqual([i for i, r in enumerate(out) if r is not None], list(range(8)))
        self.assertEqual([r for r in out if r and "error" in r], [])
        self.assertGreater(align._oom_count[0], 0, "the refusals were not counted as ooms")
        self.assertEqual(len(ctx.metered), 8, "each span is metered exactly once")

    def test_the_budget_is_not_lowered_by_a_refusal(self):
        """Splitting was never the problem; remembering was.

        A ceiling that only ever fell recovered correctly -- 4000 spans, zero errors --
        and left the process at one span a call for its whole life at about a tenth of the
        throughput, with a complete response and a zero error count.
        """
        def fake_call(ctx, group, sr, out):
            if len(group) > 2:
                raise RuntimeError(OOM)
            for m in group:
                out[m.index] = {"language": "en", "units": []}
                align._bill(ctx, m)

        align._budget = 12345.0
        self._run(fake_call)
        self.assertGreaterEqual(align._budget, 12345.0,
                                "a refusal lowered the budget, which is the collapse this "
                                "path was rewritten to stop")

    def test_a_refusal_drops_the_peak_so_the_halves_can_still_be_measured(self):
        """🔴 The other half of "remember nothing", and the one that was not true."""
        resets = []
        keep = align._reset_peak
        align._reset_peak = lambda: resets.append(1)
        try:
            def fake_call(ctx, group, sr, out):
                if len(group) > 2:
                    raise RuntimeError(OOM)
                for m in group:
                    out[m.index] = {"language": "en", "units": []}
                    align._bill(ctx, m)

            self._run(fake_call)
        finally:
            align._reset_peak = keep
        self.assertTrue(resets,
                        "an out-of-memory left the process peak where the failed call put "
                        "it, so every call after it measures as zero and the correction "
                        "can never move again")

    def test_a_warm_up_that_dies_puts_the_peak_back(self):
        """`_calibrate` resets the peak and then runs a real call, so a failure after that
        leaves the mark where the warm-up died and no later call rises above it: every
        sample is discarded, the counter never leaves zero, and the grouping gate never
        opens while the log says it is waiting for a measurement that cannot arrive.
        """
        events = []
        keep = (align._reset_peak, align._align, align._state.get("model"))
        align._reset_peak = lambda: events.append("reset")
        align._state["model"] = object()          # far enough in to reach the warm-up
        def _dies(*a, **k):
            events.append("warm-up ran")
            raise RuntimeError("CUDA out of memory during warm-up")
        align._align = _dies
        try:
            align._calibrate()                     # non-fatal by design
        finally:
            align._reset_peak, align._align = keep[0], keep[1]
            if keep[2] is None:
                align._state.pop("model", None)
            else:
                align._state["model"] = keep[2]
        self.assertIn(
            "warm-up ran", events,
            "_calibrate never reached the warm-up call, so this test asserted nothing about "
            "the peak. It dies before that when soundfile or numpy will not import -- "
            "install them on this host rather than lowering the assertion below, which is "
            "what makes the difference invisible. Events: %r" % (events,))
        self.assertEqual(
            events[-1], "reset",
            "the warm-up reset the peak, raised, and left it up there -- so nothing "
            "measurable can happen again and the grouping gate stays shut for good. "
            "Events: %r" % (events,))

    def test_any_failure_drops_the_peak_not_just_one_we_recognised(self):
        """Tied to the failure, not to its classification.

        `_is_oom` matches "out of memory" in the text, and a real memory wall that surfaces
        as CUDNN_STATUS_ALLOC_FAILED is not that string.
        """
        resets = []
        keep = align._reset_peak
        align._reset_peak = lambda: resets.append(1)
        try:
            def fake_call(ctx, group, sr, out):
                if len(group) > 1:
                    raise RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED")
                for m in group:
                    out[m.index] = {"language": "en", "units": []}
                    align._bill(ctx, m)

            self._run(fake_call)
        finally:
            align._reset_peak = keep
        self.assertTrue(resets,
                        "a memory wall that did not say 'out of memory' left the peak "
                        "where it was, so the correction freezes exactly as it did before")

    def test_the_wrapper_every_call_goes_through_resets_on_any_failure(self):
        """At the wrapper, not at a path.

        The tests above each hold ONE route to the model, and the enumeration of routes was
        wrong four times running -- three, then five, then seven, and the eighth was found
        by listing callers. tests/test_contract.py holds the other half: that nothing
        reaches the raw body and skips the wrapper altogether.
        """
        keep_rp, keep_raw = align._reset_peak, align._align_raw
        for blew_up in (RuntimeError(OOM),
                        RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"),
                        KeyboardInterrupt()):
            resets = []
            align._reset_peak = lambda: resets.append(1)

            def _raise(*a, **k):
                raise blew_up

            align._align_raw = _raise
            try:
                with self.assertRaises(type(blew_up)):
                    align._align("clip.wav", "hi", "auto")
            finally:
                align._reset_peak, align._align_raw = keep_rp, keep_raw
            self.assertEqual(resets, [1],
                             "%r left the process's all-time peak where it got to, so every "
                             "call after it measures as nothing and the correction freezes"
                             % blew_up)

        resets = []
        align._reset_peak = lambda: resets.append(1)
        align._align_raw = lambda *a, **k: [[]]
        try:
            self.assertEqual(align._align("clip.wav", "hi", "auto"), [[]])
        finally:
            align._reset_peak, align._align_raw = keep_rp, keep_raw
        self.assertEqual(resets, [],
                         "a successful call put the peak back, which discards the envelope "
                         "the caller is about to measure against")

    def test_the_peak_written_down_is_the_one_from_before_the_reset(self):
        """The test below stubs `_call`, so the real reset never runs and the figure it checks
        is whatever `_peak_bytes` was stubbed to. This one drives a real counter through the
        real wrapper: a call that climbed to 9 GB was reporting 1 kB, and that number is what
        an operator reads to set --gpu-budget-fraction.
        """
        card = {"alloc": 1000, "peak": 1000}
        keep = (align._memory_now, align._peak_bytes, align._reset_peak, align._align_raw)
        align._memory_now = lambda: (card["alloc"], card["peak"])
        align._peak_bytes = lambda: card["peak"]
        align._reset_peak = lambda: card.__setitem__("peak", card["alloc"])

        def climbs_then_fails(*a, **k):
            card["peak"] = 9_000_000          # where the card actually refused
            raise RuntimeError(OOM)

        align._align_raw = climbs_then_fails
        try:
            with self.assertRaises(RuntimeError):
                align._align("clip.wav", "hi", "auto")
            self.assertEqual(
                align._peak_at_failure[0], 9_000_000,
                "the high-water mark was read after the reset, so the refused call's row "
                "carries the rebased counter instead of the wall it hit")
            self.assertEqual(card["peak"], card["alloc"],
                             "the reset did not happen, which is the other half")
            align._align_raw = lambda *a, **k: [[]]
            align._align("clip.wav", "hi", "auto")
            self.assertIsNone(align._peak_at_failure[0],
                              "a successful call left the previous failure's peak in place")
        finally:
            (align._memory_now, align._peak_bytes, align._reset_peak,
             align._align_raw) = keep
            align._peak_at_failure[0] = None

    def test_what_the_reset_throws_away_is_written_down_first(self):
        """`_used_bytes` forbids resetting inside a request because the bench frames one with
        the same counter. The reset happens anyway, so what it discards goes onto the
        group's own row instead of vanishing.
        """
        keep = (align._reset_peak, align._peak_bytes)
        align._reset_peak = lambda: None
        align._peak_bytes = lambda: 4242
        try:
            def fake_call(ctx, group, sr, out):
                if len(group) > 2:
                    raise RuntimeError(OOM)
                for m in group:
                    out[m.index] = {"language": "en", "units": []}
                    align._bill(ctx, m)

            self._run(fake_call)
        finally:
            align._reset_peak, align._peak_bytes = keep
        failed = [g for g in align._groups if g["failed"]]
        self.assertTrue(failed, "no group recorded a failure, so nothing was pinned")
        self.assertEqual([g.get("peak_bytes") for g in failed], [4242] * len(failed),
                         "the peak a refused call reached was discarded, not recorded")

    def test_the_ramp_converges_instead_of_running_away(self):
        """The property a ceiling was briefly added here to enforce, and which holds without
        one: the ramp raises the budget to twice THIS group's cost, and a group cannot
        exceed the request carrying it, so it converges rather than running away.
        """
        keep = (align._solve_budget, align._budget, align._call)
        try:
            align._solve_budget = lambda: None          # neither authority readable

            def fake_call(ctx, group, sr, out):
                for m in group:
                    out[m.index] = {"language": "en", "units": []}
                    align._bill(ctx, m)

            align._call = fake_call
            align._budget = 1.0
            spans = _spans(64, seconds=30.0, positions=400)
            settled = []
            for _ in range(6):                          # six identical requests
                align._billed.clear()
                at, out, ctx = 0, [None] * 64, _Ctx()
                while at < len(spans):
                    g, at, why = align._next_group(spans, list(range(64)), at)
                    align._run_group(ctx, g, 16000, out, why)
                settled.append(align._budget)

            self.assertEqual(
                settled[-1], settled[-3],
                "the budget was still moving after six identical requests, so the ramp is "
                "not converging: %r" % (settled,))
            self.assertLessEqual(
                settled[-1], 2.0 * align._cost(spans),
                "the budget settled above twice the whole request's cost, which is the "
                "bound the climb is supposed to have: %r" % (settled[-1],))
        finally:
            align._solve_budget, align._budget, align._call = keep

    def test_one_unreadable_span_does_not_cost_the_others_theirs(self):
        def fake_call(ctx, group, sr, out):
            if any(m.index == 3 for m in group) and len(group) > 1:
                raise RuntimeError("could not read the audio")
            for m in group:
                if m.index == 3:
                    raise RuntimeError("could not read the audio")
                out[m.index] = {"language": "en", "units": []}
                align._bill(ctx, m)

        ctx, out, done = self._run(fake_call)
        self.assertEqual(done, 8)
        self.assertEqual([i for i, r in enumerate(out) if r and "error" in r], [3])
        self.assertEqual(align._oom_count[0], 0,
                         "a failure that was not about size was counted as one")

    def test_a_span_already_answered_is_not_retried_and_cannot_be_undone(self):
        """`_call` writes and bills each span as it comes off the batch, so a raise partway
        through leaves some already answered. Retrying those would bill them twice and
        could overwrite a good answer with an error.
        """
        attempts = []

        def fake_call(ctx, group, sr, out):
            attempts.append([m.index for m in group])
            if len(group) > 1:
                for m in group[:2]:
                    out[m.index] = {"language": "en", "units": [{"ok": m.index}]}
                    align._bill(ctx, m)
                raise RuntimeError("could not read the result for span %d" % group[2].index)
            raise RuntimeError("could not read the result for span %d" % group[0].index)

        ctx, out, done = self._run(fake_call, n=6)
        self.assertEqual(done, 6)
        self.assertEqual(out[0], {"language": "en", "units": [{"ok": 0}]},
                         "a span that had already succeeded was retried and lost its result")
        self.assertEqual(out[1], {"language": "en", "units": [{"ok": 1}]})
        self.assertTrue(all(out[i] and "error" in out[i] for i in range(2, 6)))
        self.assertNotIn([0], attempts, "span 0 was answered and must not be called again")
        self.assertNotIn([1], attempts)

    def test_a_span_that_fails_alone_comes_back_as_that_span_s_error(self):
        def fake_call(ctx, group, sr, out):
            raise RuntimeError(OOM)

        ctx, out, done = self._run(fake_call, n=4)
        self.assertEqual(done, 4)
        self.assertEqual(len(out), 4)
        self.assertTrue(all(r and "error" in r for r in out))

    def test_recursion_is_not_how_this_works(self):
        """A stack, not recursion. Depth was never the risk -- nesting the handlers was."""
        import inspect

        body = inspect.getsource(align._run_group).split("\n", 1)[1]
        self.assertNotIn("_run_group(", body,
                         "_run_group calls itself again, so each retry nests inside the "
                         "previous failure's handler")


class ExhaustedGrantTest(unittest.TestCase):
    """A grant that read and has nothing left is not a grant that could not be read."""

    def setUp(self):
        self._keep = (align._budget, align._budget_priced, align._opening_logged,
                      align._solve_budget, align.ALIGN_FIXED_BUDGET)
        self.addCleanup(self._restore)

    def _restore(self):
        (align._budget, align._budget_priced, align._opening_logged,
         align._solve_budget, align.ALIGN_FIXED_BUDGET) = self._keep

    def _say(self, solved):
        align._solve_budget = lambda: solved
        align.ALIGN_FIXED_BUDGET = 0.0
        align._budget, align._opening_logged = 0.0, False
        with self.assertLogs(align.log, level="WARNING") as got:
            align._effective_budget()
        return "\n".join(got.output)

    def test_a_readable_grant_with_nothing_left_is_not_reported_as_unreadable(self):
        """`_next_group` calls this state `closed_by: memory`, so the log saying the card is
        full has to mean the card, not an authority nobody could read.
        """
        line = self._say(0.0)
        self.assertIn("nothing left", line)
        self.assertNotIn("could not be read", line)

    def test_an_unreadable_grant_still_says_so(self):
        line = self._say(None)
        self.assertIn("could not be read", line)


class HamiLimitTest(unittest.TestCase):
    """What HAMi enforces, read the way HAMi reads it."""

    def setUp(self):
        import os

        self._env = {k: os.environ.get(k) for k in
                     ("CUDA_DEVICE_MEMORY_LIMIT_0", "CUDA_DEVICE_MEMORY_LIMIT")}
        align._hami_unreadable_logged = False
        self.addCleanup(self._restore)

    def _restore(self):
        import os

        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        align._hami_unreadable_logged = False

    def _set(self, indexed=None, plain=None):
        import os

        for k, v in (("CUDA_DEVICE_MEMORY_LIMIT_0", indexed),
                     ("CUDA_DEVICE_MEMORY_LIMIT", plain)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_the_unindexed_variable_is_read_when_there_is_no_per_device_one(self):
        """`do_init_device_memory_limits` takes `CUDA_DEVICE_MEMORY_LIMIT` as the fallback when
        the per-device variable is unset, so a deployment setting only the unindexed one was
        being sized against a limit HAMi was enforcing and this engine could not see.
        """
        self._set(indexed=None, plain="4096m")
        self.assertEqual(align._hami_limit_bytes(), 4096 * 2 ** 20)

    def test_the_per_device_one_wins(self):
        self._set(indexed="2048m", plain="4096m")
        self.assertEqual(align._hami_limit_bytes(), 2048 * 2 ** 20)

    def test_neither_set_is_no_limit(self):
        self._set()
        self.assertEqual(align._hami_limit_bytes(), 0)

    def test_a_leading_zero_is_octal_because_hami_parses_base_zero(self):
        """`strtoul(s, end, 0)`: `"04096m"` is 040 octal -- strtoul stops at the 9 -- so HAMi
        enforces 32 MiB, and reading it as 4096 would size every call against a figure 128
        times too large. Matched deliberately, not by accident.
        """
        self._set(indexed="04096m")
        self.assertEqual(align._hami_limit_bytes(), 32 * 2 ** 20)
        self._set(indexed="0x10m")
        self.assertEqual(align._hami_limit_bytes(), 16 * 2 ** 20)


class DimsWarningTest(unittest.TestCase):
    """The self-check that says the cost model is not running must survive being scraped."""

    def setUp(self):
        self._model = align._state.get("model")
        align._dims_unreadable = False
        self.addCleanup(self._restore)

    def _restore(self):
        align._state["model"] = self._model
        align._dims_unreadable = False

    def test_reading_dims_before_the_model_loads_says_nothing(self):
        """A monitor discovers /v1/audio/align/telemetry -- it is advertised in the catalog --"""
        align._state["model"] = None
        with self.assertNoLogs(align.log, level="WARNING"):
            self.assertIsNone(align._dims())
        self.assertFalse(align._dims_unreadable,
                         "a startup scrape spent the one-shot flag, so the real condition "
                         "arriving later would print nothing at all")

    def test_a_loaded_checkpoint_whose_dims_will_not_read_still_warns(self):
        """And it still fires when it should, or the guard above turned the self-check off"""
        align._state["model"] = object()          # loaded, and nothing readable on it
        with self.assertLogs(align.log, level="WARNING") as got:
            self.assertIsNone(align._dims())
        self.assertIn("no cost model", "\n".join(got.output))
        self.assertTrue(align._dims_unreadable)


class NothingAlignedTest(unittest.TestCase):
    def test_every_span_failing_is_logged_at_error(self):
        results = [{"error": "align failed: " + OOM} for _ in range(3)]
        with self.assertLogs(align.log, level="ERROR") as got:
            align._say_if_nothing_aligned(results)
        self.assertIn("not one of 3 spans aligned", "\n".join(got.output))

    def test_a_partial_failure_stays_quiet(self):
        """Ordinary, and shouting about it would train whoever watches the log to skip the"""
        results = [{"units": []}, {"error": "align failed: could not read the audio"}]
        with self.assertNoLogs(align.log, level="ERROR"):
            align._say_if_nothing_aligned(results)


class SpanCeilingTest(unittest.TestCase):
    """🔴 The one limit whose failure is a wrong answer rather than a missing one."""

    @staticmethod
    def _with(engine_args):
        import importlib
        import os
        import sys
        was = os.environ.get("ENGINE_ARGS")
        os.environ["ENGINE_ARGS"] = engine_args
        try:
            for name in [m for m in sys.modules if m.startswith("wrapper")]:
                del sys.modules[name]
            return importlib.import_module("wrapper.caps.align")
        finally:
            if was is None:
                os.environ.pop("ENGINE_ARGS", None)
            else:
                os.environ["ENGINE_ARGS"] = was

    def test_a_smaller_head_actually_lowers_the_ceiling(self):
        """The reason the whole mechanism exists, and until now it had never run.

        The shipped head agrees with the hand-written constant, so `_rebind_span_limit`
        returns on its first line in every other test and the code that lowers the limit was
        dead on every run.
        """
        mod = self._with("")
        keep = (mod.EXPRESSIBLE_SPAN_SEC, mod.MAX_SPAN_SEC)
        try:
            mod._rebind_span_limit({"classes": 2000, "grid_ms": 80.0})
            self.assertEqual(mod.EXPRESSIBLE_SPAN_SEC, 160.0,
                             "the ceiling did not follow the checkpoint, so a model with a "
                             "shorter head would go on being asked for spans it answers "
                             "wrongly rather than refusing them")
            self.assertEqual(mod.MAX_SPAN_SEC, 160.0,
                             "the ceiling moved but the limit actually enforced did not, "
                             "which is the same wrong answer with a correct-looking constant")
        finally:
            mod.EXPRESSIBLE_SPAN_SEC, mod.MAX_SPAN_SEC = keep

    def test_a_coarser_grid_lowers_it_too_and_a_bigger_head_raises_nothing_by_itself(self):
        """⚠️ Two halves. The grid is the other input and moves the answer on its own."""
        mod = self._with("--max-span-seconds 120")
        keep = (mod.EXPRESSIBLE_SPAN_SEC, mod.MAX_SPAN_SEC)
        try:
            mod._rebind_span_limit({"classes": 5000, "grid_ms": 40.0})   # 200 s
            self.assertEqual(mod.EXPRESSIBLE_SPAN_SEC, 200.0)
            self.assertEqual(mod.MAX_SPAN_SEC, 120.0,
                             "the checkpoint overrode what the deployment asked for")
            mod._rebind_span_limit({"classes": 20000, "grid_ms": 80.0})  # 1600 s
            self.assertEqual(mod.MAX_SPAN_SEC, 120.0,
                             "a longer head raised a limit nobody raised")
        finally:
            mod.EXPRESSIBLE_SPAN_SEC, mod.MAX_SPAN_SEC = keep

    def test_dimensions_that_will_not_read_keep_the_limit_rather_than_losing_it(self):
        mod = self._with("")
        keep = (mod.EXPRESSIBLE_SPAN_SEC, mod.MAX_SPAN_SEC)
        try:
            for bad in ({}, None, {"classes": 0, "grid_ms": 80.0}, {"classes": 5000}):
                mod._rebind_span_limit(bad)
                self.assertEqual(mod.EXPRESSIBLE_SPAN_SEC, keep[0], bad)
                self.assertEqual(mod.MAX_SPAN_SEC, keep[1], bad)
        finally:
            mod.EXPRESSIBLE_SPAN_SEC, mod.MAX_SPAN_SEC = keep

    def test_the_ceiling_is_the_head_arithmetic(self):
        mod = self._with("")
        self.assertEqual(mod.EXPRESSIBLE_SPAN_SEC, 5000 * 0.080)

    def test_asking_past_the_head_is_clamped_and_said(self):
        with self.assertLogs("audio-align", level="WARNING") as got:
            mod = self._with("--max-span-seconds 600")
        self.assertEqual(mod.MAX_SPAN_SEC, mod.EXPRESSIBLE_SPAN_SEC)
        self.assertIn("--max-span-seconds", "\n".join(got.output))

    def test_lowering_it_is_left_alone(self):
        self.assertEqual(self._with("--max-span-seconds 120").MAX_SPAN_SEC, 120.0)

    def test_the_shipped_default_is_under_the_ceiling(self):
        mod = self._with("")
        self.assertEqual(mod.MAX_SPAN_SEC, 300.0)
        self.assertLess(mod.MAX_SPAN_SEC, mod.EXPRESSIBLE_SPAN_SEC)

    def test_the_refusal_says_the_whole_sentence(self):
        """The exact wording, because it is published as interface: a caller matching on the
        text breaks when it moves."""
        mod = self._with("")
        self.assertEqual(
            mod._admit(310.0, "anything"),
            "span is 310.0s; this deployment aligns at most 300s (--max-span-seconds)")


if __name__ == "__main__":
    unittest.main()
