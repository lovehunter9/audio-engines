"""How a batch gets sized: the calibration, the correction, and the budget.

🔴 Every test here is about a way the sizing can be WRONG WITHOUT SAYING SO. A cost model
that is never checked, a calibration that measured nothing while reporting a number, a
grant that could not be read reported as a grant with nothing left -- all three serve
traffic and all three look like a working engine from outside.
"""
import importlib
import os
import sys
import types
import io
import unittest
from unittest import mock


def _stt(engine_args=""):
    env = {"AUDIO_BASE": "qwen", "MODEL_SUPPORTS": "stt,stt_stream", "ENGINE_ARGS": engine_args}
    with mock.patch.dict(os.environ, env, clear=False):
        return importlib.reload(importlib.import_module("wrapper.caps.stt_stream"))


class UsedTest(unittest.TestCase):
    """The rule that decides whether a reading is a measurement at all."""

    def test_a_call_that_did_not_raise_the_peak_measured_nothing(self):
        # 🔴 The counter is the process's all-time high. A call under an older peak adds
        # nothing to it, and answering with a difference anyway invents a measurement.
        m = _stt()
        self.assertEqual(m._used(before=100, peak_before=500, peak_after=500), 0)
        self.assertEqual(m._used(before=100, peak_before=500, peak_after=400), 0)

    def test_a_call_that_raised_the_peak_is_measured_from_where_it_started(self):
        m = _stt()
        self.assertEqual(m._used(before=100, peak_before=500, peak_after=900), 800)

    def test_the_rule_is_readable_without_a_card(self):
        # The reading and the rule failed together in the aligner's version, which returned
        # a number whichever way the comparison went. Split apart, the rule is checkable.
        m = _stt()
        self.assertEqual(m._used(0, 0, 0), 0)


class ObserveTest(unittest.TestCase):
    def test_the_correction_is_measured_over_predicted(self):
        m = _stt()
        padded = 60.0
        m._observe(padded, 2 * padded * m.OPENING_BYTES_A_PADDED_SECOND)
        self.assertAlmostEqual(m._scale, 2.0)
        self.assertEqual((m._scale_seen, m._calls_seen), (1, 1))

    def test_a_call_that_measured_nothing_still_counts_as_a_call(self):
        # 🔴 The two counters exist to keep these apart: a call that taught nothing is not
        # a call that did not happen, and the ratio that did arrive is the upper envelope.
        m = _stt()
        m._observe(60.0, 0)
        self.assertEqual((m._scale_seen, m._calls_seen), (0, 1))
        self.assertAlmostEqual(m._scale, 1.0)

    def test_the_smoothing_follows_a_run_of_calls_without_chasing_one(self):
        m = _stt()
        per = m.OPENING_BYTES_A_PADDED_SECOND
        m._observe(60.0, 60.0 * per)          # exactly as predicted
        m._observe(60.0, 60.0 * per * 3.0)    # one odd call
        self.assertLess(m._scale, 3.0, "one call moved the whole correction")
        self.assertGreater(m._scale, 1.0, "one call moved nothing at all")

    def test_leaving_the_band_is_said_once_not_once_a_call(self):
        # 🔴 Per call. A machine outside the band emits one line per group of a forty
        # minute meeting, burying the one-shot lines the sizing depends on being readable.
        m = _stt()
        said = []
        with mock.patch.object(m, "_p", said.append):
            for _ in range(m.SCALE_MEMORY * 3):
                m._observe(60.0, 60.0 * m.OPENING_BYTES_A_PADDED_SECOND * 10.0)
        self.assertEqual(len([x for x in said if "off by" in x]), 1, said)

    def test_a_correction_inside_the_band_says_nothing(self):
        # 🔴 A band with nothing inside it warns on every call -- the same as no band --
        # and every assertion about the warning still passes.
        m = _stt()
        said = []
        with mock.patch.object(m, "_p", said.append):
            for _ in range(m.SCALE_MEMORY):
                m._observe(60.0, 60.0 * m.OPENING_BYTES_A_PADDED_SECOND * 1.3)
        self.assertEqual(said, [], "a correction of 1.3x is not worth a line")

    def test_leaving_the_band_says_the_shape_is_wrong_and_names_both_counters(self):
        m = _stt()
        said = []
        with mock.patch.object(m, "_p", said.append):
            for _ in range(m.SCALE_MEMORY):
                m._observe(60.0, 60.0 * m.OPENING_BYTES_A_PADDED_SECOND * 10.0)
        warned = [line for line in said if "off by" in line]
        self.assertTrue(warned, "a correction of 10x said nothing")
        # 🔴 Both counters in the line. "1.12 over 7" was written up as the model being 12%
        # low, which a sample of upper-envelope ratios cannot say.
        self.assertIn("of", warned[-1])
        self.assertIn("calls", warned[-1])
        self.assertIn("padded", warned[-1])


class ScaleMemoryTest(unittest.TestCase):
    """How fast the correction forgets, which nothing was holding."""

    def test_the_weight_stops_shrinking_at_the_memory_bound(self):
        # 🔴 The cap lets an engine whose machine CHANGES under it follow: without it the
        # 50th call carries weight 1/50, with it every call past the eighth carries 1/8.
        m = _stt()
        unit = float(m.OPENING_BYTES_A_PADDED_SECOND)
        for _ in range(m.SCALE_MEMORY + 5):
            m._observe(1.0, unit)                # ratio 1.0, over and over
        self.assertAlmostEqual(m._scale, 1.0, places=6)
        m._observe(1.0, 3.0 * unit)              # the machine changed: ratio 3.0
        # An eighth of the way from 1.0 to 3.0, not a fourteenth of it.
        self.assertAlmostEqual(m._scale, 1.0 + 2.0 / m.SCALE_MEMORY, places=6)


class HeadroomTest(unittest.TestCase):
    def test_no_declared_grant_means_no_budget_rather_than_the_whole_card(self):
        """🔴 The safety property, and it is the one a later change would quietly undo.

        What CUDA reports free is this process's slice only where the sharing layer
        virtualizes the memory view. Where it does not, "free" is what the neighbours have
        not claimed yet, and a caching allocator does not give back what it takes. From
        inside, with no grant to compare against, the two cases are indistinguishable --
        so no grant means no automatic sizing.
        """
        m = _stt()
        m._resting_bytes = 0
        asked = []
        # 🔴 The container is stubbed readable ON PURPOSE: the host account must not invent
        # a GPU budget where there is no GPU authority, and an absent cgroup cannot say so.
        cg = mock.patch.object(m.cgroup, "read",
                               lambda: {"current": 0, "max": 8 * (2 ** 30),
                                        "available": None, "reclaimable": 0})
        self.addCleanup(cg.stop)
        cg.start()
        # 🔴 A stub that answers every counter, so this reaches the branch it is about.
        # Leaving `memory_allocated` off makes `_headroom_bytes` return None one line early.
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            memory_allocated=lambda: 0,
            memory_reserved=lambda: 0,
            mem_get_info=lambda: asked.append(True) or (8 * (2 ** 30), 16 * (2 ** 30)))
        with mock.patch.dict(os.environ, {"REQUIRED_GPU_MEMORY": "",
                                          "CUDA_DEVICE_MEMORY_LIMIT_0": "",
                                          "CUDA_DEVICE_MEMORY_LIMIT": ""}), \
                mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}):
            self.assertIsNone(m._headroom_bytes())
            self.assertEqual(asked, [], "it asked the card how much was free")

    def test_nothing_measured_at_rest_is_not_a_grant_of_zero(self):
        m = _stt()
        m._resting_bytes = None
        with mock.patch.dict(os.environ, {"REQUIRED_GPU_MEMORY": "12Gi"}):
            self.assertIsNone(m._headroom_bytes())


class HamiLimitTest(unittest.TestCase):
    """The ENFORCED limit, read the way HAMi's own parser reads it.

    🔴 Each of these is a factor in the over-sizing direction if we normalise it away, and
    none of them would fail anything: the engine would go on serving, sized against a
    figure nothing holds it to, with only an out-of-memory counter to show for it.
    """

    def _limit(self, env):
        m = _stt()
        with mock.patch.dict(os.environ, env, clear=False):
            return m, m._hami_limit_bytes()

    def test_a_unit_letter_is_read_as_hami_reads_it(self):
        _m, v = self._limit({"CUDA_DEVICE_MEMORY_LIMIT_0": "4096m"})
        self.assertEqual(v, 4096 * (2 ** 20))

    def test_a_trailing_space_makes_the_value_bytes_not_mebibytes(self):
        # 🔴 HAMi indexes the last character, so with a trailing space "4096m " is 4096
        # BYTES. Stripping first reads it as 4 GiB -- 2**20 too big, silently.
        _m, v = self._limit({"CUDA_DEVICE_MEMORY_LIMIT_0": "4096m "})
        self.assertEqual(v, 4096)

    def test_a_leading_zero_is_octal_because_hami_parses_base_zero(self):
        # 🔴 Base zero makes the leading zero octal and 9 is not an octal digit, so strtoul
        # stops: "04096m" is 0o40 = 32 MiB to HAMi, 4096 MiB to any decimal reader. 128x.
        _m, v = self._limit({"CUDA_DEVICE_MEMORY_LIMIT_0": "04096m"})
        self.assertEqual(v, 32 * (2 ** 20))

    def test_digits_that_int_would_take_and_hami_would_not(self):
        # `"٤٠٩٦m"` is digits to str.isdigit and to int(); it is nothing to strtoul, so it
        # is no limit here. Reading it as 4 GiB would size against a figure HAMi calls zero.
        _m, v = self._limit({"CUDA_DEVICE_MEMORY_LIMIT_0": "٤٠٩٦m"})
        self.assertEqual(v, 0)

    def test_zero_is_no_limit_and_is_not_an_error(self):
        said = []
        m = _stt()
        with mock.patch.dict(os.environ, {"CUDA_DEVICE_MEMORY_LIMIT_0": "0m"}), \
                mock.patch.object(m, "_p", said.append):
            self.assertEqual(m._hami_limit_bytes(), 0)
        self.assertEqual(said, [], "0 means no limit; warning sends someone after a typo")

    def test_the_unindexed_variable_is_read_when_the_per_device_one_is_absent(self):
        # hami-core falls back to it for every device without its own. Reading only `_0`
        # makes a deployment configured that way look unenforced.
        m = _stt()
        with mock.patch.dict(os.environ, {"CUDA_DEVICE_MEMORY_LIMIT_0": "",
                                          "CUDA_DEVICE_MEMORY_LIMIT": "8g"}):
            self.assertEqual(m._hami_limit_bytes(), 8 * (2 ** 30))

    def test_a_value_hami_cannot_parse_is_no_limit_and_says_so(self):
        said = []
        m = _stt()
        with mock.patch.dict(os.environ, {"CUDA_DEVICE_MEMORY_LIMIT_0": "lots"}), \
                mock.patch.object(m, "_p", said.append):
            self.assertEqual(m._hami_limit_bytes(), 0)
        self.assertIn("no number at all", " ".join(said))


class HeadroomAuthorityTest(unittest.TestCase):
    """Which authority a batch is sized against, in each of the four states."""

    def _module(self, total=None, free=None, allocated=2 * (2 ** 30), reserved=None):
        """A module whose card reports `total`/`free`, and which holds `allocated`.

        🔴 `reserved` defaults to `allocated`, i.e. no cache. The cache term is what the
        interposer has already charged us and we can spend again without asking, so a stub
        that leaves it out tests the arithmetic with that half removed.
        """
        m = _stt()
        m._resting_bytes = allocated
        cuda = types.SimpleNamespace(
            is_available=lambda: True,
            memory_allocated=lambda: allocated,
            memory_reserved=lambda: (allocated if reserved is None else reserved),
            mem_get_info=lambda: (free, total))
        return m, mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)})

    def test_an_enforced_limit_with_rewritten_counters_uses_what_they_report(self):
        # total at or under the published limit can only come from the interposer's
        # rewrite, and then `free` is this container's own arithmetic.
        m, torched = self._module(total=8 * (2 ** 30), free=5 * (2 ** 30))
        with mock.patch.dict(os.environ, {"CUDA_DEVICE_MEMORY_LIMIT_0": "8g"}), torched:
            self.assertEqual(m._headroom_bytes(), 5 * (2 ** 30))

    def test_blocks_already_bought_and_freed_are_spendable_again(self):
        # 🔴 HAMi charges the allocation, not the use. What torch bought and freed can be
        # spent without asking it, so leaving it out shrinks the budget as the process runs.
        m, torched = self._module(total=8 * (2 ** 30), free=5 * (2 ** 30),
                                  allocated=1 * (2 ** 30), reserved=3 * (2 ** 30))
        with mock.patch.dict(os.environ, {"CUDA_DEVICE_MEMORY_LIMIT_0": "8g"}), torched:
            self.assertEqual(m._headroom_bytes(), 5 * (2 ** 30) + 2 * (2 ** 30))

    def test_an_enforced_limit_whose_counters_show_the_whole_card_is_not_believed(self):
        # 🔴 The case that would spend a neighbour's memory: the limit is enforced but the
        # counters were not rewritten, so `free` is the whole card.
        m, torched = self._module(total=80 * (2 ** 30), free=60 * (2 ** 30))
        with mock.patch.dict(os.environ, {"CUDA_DEVICE_MEMORY_LIMIT_0": "8g"}), torched:
            self.assertEqual(m._headroom_bytes(), 8 * (2 ** 30) - 2 * (2 ** 30))

    def test_no_enforced_limit_falls_back_to_the_declared_one(self):
        m, torched = self._module(total=80 * (2 ** 30), free=60 * (2 ** 30))
        with mock.patch.dict(os.environ, {"CUDA_DEVICE_MEMORY_LIMIT_0": "",
                                          "CUDA_DEVICE_MEMORY_LIMIT": "",
                                          "REQUIRED_GPU_MEMORY": "12Gi"}), torched:
            self.assertEqual(m._headroom_bytes(), 10 * (2 ** 30))

    def test_with_no_card_there_is_no_cache_to_add_back(self):
        # 🔴 Zero, not "some". The cache term is added to a headroom, so anything else here
        # would hand a machine with no card a budget out of nothing.
        m = _stt()
        cuda = types.SimpleNamespace(is_available=lambda: False)
        with mock.patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)}):
            self.assertEqual(m._cached_bytes(), 0)
            self.assertIsNone(m._held_bytes())
        with mock.patch.dict(sys.modules, {"torch": None}):
            self.assertEqual(m._cached_bytes(), 0)

    def test_holding_more_than_the_grant_is_no_room_rather_than_negative_room(self):
        # 🔴 Reachable: a quota is a declaration and nothing enforces it, so this process
        # can be over it. "Less than nothing" is not a size anything downstream takes.
        m, torched = self._module(total=80 * (2 ** 30), free=60 * (2 ** 30),
                                  allocated=20 * (2 ** 30))
        with mock.patch.dict(os.environ, {"CUDA_DEVICE_MEMORY_LIMIT_0": "8g"}), torched:
            self.assertEqual(m._headroom_bytes(), 0)
        with mock.patch.dict(os.environ, {"CUDA_DEVICE_MEMORY_LIMIT_0": "",
                                          "CUDA_DEVICE_MEMORY_LIMIT": "",
                                          "REQUIRED_GPU_MEMORY": "12Gi"}), torched:
            self.assertEqual(m._headroom_bytes(), 0)

    def test_neither_authority_is_no_budget_at_all(self):
        m, torched = self._module(total=80 * (2 ** 30), free=60 * (2 ** 30))
        with mock.patch.dict(os.environ, {"CUDA_DEVICE_MEMORY_LIMIT_0": "",
                                          "CUDA_DEVICE_MEMORY_LIMIT": "",
                                          "REQUIRED_GPU_MEMORY": ""}), torched:
            self.assertIsNone(m._headroom_bytes())


class BudgetTest(unittest.TestCase):
    def _auto(self, headroom):
        m = _stt()
        m._resting_bytes = 0
        return m, mock.patch.object(m, "_headroom_bytes", lambda: headroom)

    def test_a_readable_grant_becomes_padded_seconds(self):
        m, headroom = self._auto(12 * (2 ** 30))
        with headroom:
            budget = m._solve_budget()
        # Half of 12 GiB at the opening 6 MiB a padded second.
        self.assertAlmostEqual(budget, 0.5 * 12 * 1024 / 6.0, places=3)

    def test_a_refusal_outranks_the_arithmetic(self):
        # 🔴 The one reading the model cannot make for itself: a call that died allocated
        # nothing to measure, so the refusal is the only evidence the prediction was high.
        m, headroom = self._auto(12 * (2 ** 30))
        m._note_refusal(100.0)
        with headroom:
            self.assertEqual(m._solve_budget(), 50.0)

    def test_the_two_zeros_are_two_different_sentences(self):
        # 🔴 "could not be read" sends an operator after a fault that is not there;
        # "read and spent" is the card being full, and it lifts by itself.
        unreadable, spent = [], []
        m, headroom = self._auto(None)
        with headroom, mock.patch.object(m, "_p", unreadable.append):
            self.assertIsNone(m._effective_budget())
        m, headroom = self._auto(0)
        with headroom, mock.patch.object(m, "_p", spent.append):
            self.assertEqual(m._effective_budget(), 0.0)
        self.assertNotEqual(unreadable, spent)
        self.assertIn("nothing to size from", " ".join(unreadable))
        self.assertIn("already spent", " ".join(spent))

    def test_the_budget_is_said_once_not_once_a_group_and_says_the_number(self):
        m, headroom = self._auto(12 * (2 ** 30))
        said = []
        with headroom, mock.patch.object(m, "_p", said.append):
            for _ in range(5):
                m._effective_budget()
        self.assertEqual(len(said), 1, "a 40 minute meeting is hundreds of groups")
        # The number itself, not just that a line appeared: the healthy branch and the
        # "grant already spent" branch are one comparison apart.
        self.assertIn("1024 padded audio seconds", said[0])

    def test_no_flag_can_put_a_number_where_nothing_can_be_measured(self):
        # 🔴 This used to be the opposite: `--batch-max-seconds 600` stood in for a machine that
        # could not be measured. There is no such flag now, on purpose -- a size this engine
        # cannot check against anything is a size nobody should be able to write, and the
        # deployment that thinks it knows better says so by declaring REQUIRED_GPU_MEMORY.
        m = _stt("--batch-max-spans 32")
        with mock.patch.object(m, "_headroom_bytes", lambda: None), \
             mock.patch.object(m.cgroup, "read",
                               lambda: {"current": None, "max": None, "available": None}):
            self.assertIsNone(m._effective_budget())

    def test_the_budget_is_what_the_machine_measures(self):
        # 🔴 What stood here was "a configured number is lowered by the measurement", from a
        # surface that had a number to configure. There is none; the measurement is the whole
        # answer, and this is the assertion on `_effective_budget` rather than `_solve_budget`.
        m = _stt()
        m._resting_bytes = 0
        with mock.patch.object(m, "_headroom_bytes", lambda: 2400 * (2 ** 20)), \
                mock.patch.object(m.cgroup, "read",
                                  lambda: {"current": None, "max": None, "available": None}):
            self.assertAlmostEqual(m._effective_budget(), 200.0, places=3)

    def test_a_zero_length_span_answers_with_language_on_both_paths(self):
        # 🔴 One shape on both paths. A client reading results[i]["language"] got a KeyError
        # from whichever path ran, and which one runs changed when the default became `auto`.
        import re as _re
        src = io.open("wrapper/caps/stt_stream.py", encoding="utf-8").read()
        empties = _re.findall(r'\{"text": ""[^}]*\}', src)
        self.assertTrue(empties, "no empty-span answer found; this test lost its target")
        for shape in empties:
            self.assertIn('"language"', shape, shape)

    def test_a_span_ceiling_does_not_turn_the_seconds_budget_off(self):
        """🔴 The one way a flag could make a call BIGGER, and it is closed.

        What stood here asserted the opposite -- that a count meant no seconds budget --
        which was true of a surface that had a count MODE forming groups on its own. That
        mode is gone; a count is a ceiling on top of the measurement. ⚠️ And the old
        assertion passed for a reason of its own: `_effective_budget` answers None on any
        machine with no memory counter, which is every machine that runs this file, so it
        would have passed with no flags at all. The stub below is what makes it an assertion.
        """
        m = _stt("--batch-max-spans 32")
        m._resting_bytes = 0
        with mock.patch.object(m, "_headroom_bytes", lambda: 12288 * (2 ** 20)), \
                mock.patch.object(m.cgroup, "read",
                                  lambda: {"current": None, "max": None, "available": None}):
            self.assertAlmostEqual(m._effective_budget(), 0.5 * 12288 / 6.0, places=3)
        self.assertEqual(m.MAX_SPANS, 32)

class FlagTest(unittest.TestCase):
    """One flag, and what it does when it cannot be read.

    🔴 `--batch-max-spans N` is the only batching flag, and it answers the only question an
    operator is better placed to answer than this engine: how many spans a call may carry, as a
    backstop for when the sizing is wrong. How much MEMORY a call may take is declared once at
    install as REQUIRED_GPU_MEMORY. How the spans inside the count are grouped needs the grant,
    the container and the neighbours on the card, which are invisible from outside the process.
    """

    def _said(self, engine_args):
        m = _stt(engine_args)
        said = []
        with mock.patch.object(m, "_p", said.append):
            m._say_batching()
        return m, " ".join(said)

    def test_nothing_given_is_on_with_no_ceiling_and_says_so(self):
        m, said = self._said("")
        self.assertEqual((m.GROUPING, m.MAX_SPANS), (True, None))
        self.assertIn("sized from what this machine measures", said)
        self.assertIn("`--batch-max-spans 1` is one span a call", said)

    def test_one_is_how_batching_is_turned_off(self):
        # 🔴 The count IS the switch. A separate on/off flag would be a second way to say a
        # thing this one already says, and two ways to say it are two things to disagree.
        m, said = self._said("--batch-max-spans 1")
        self.assertEqual((m.GROUPING, m.MAX_SPANS), (False, 1))
        self.assertIn("one span a call", said)

    def test_a_count_above_one_leaves_the_sizing_to_the_measurement(self):
        m, said = self._said("--batch-max-spans 32")
        self.assertEqual((m.GROUPING, m.MAX_SPANS), (True, 32))
        self.assertIn("never more than 32 spans", said)

    def test_an_unreadable_count_is_no_ceiling_and_says_so(self):
        # 🔴 No ceiling rather than a guessed one, and said: a misspelled flag otherwise reaches
        # the engine as the default it fell back to, and a default is what working looks like.
        for bad in ("32x", "0", "-1", "", "1.5"):
            m, said = self._said("--batch-max-spans %s" % bad)
            self.assertIsNone(m.MAX_SPANS, bad)
            self.assertTrue(m.GROUPING, bad)
            self.assertIn("WARN", said, bad)

    def test_the_flags_this_engine_used_to_take_are_gone(self):
        # ⚠️ `--batch` and `--batch-max-seconds` existed on this branch while the surface was
        # being settled and neither shipped. Written today they are unclaimed flags, which the
        # contract reports on its own; nothing here reads them.
        m = _stt("--batch auto --batch-max-seconds 600")
        self.assertEqual((m.GROUPING, m.MAX_SPANS), (True, None))


class BudgetShareTest(unittest.TestCase):
    """The safety margin, which is a constant on purpose."""

    def _budget(self, args, grant_mib=12288):
        m = _stt(args)
        m._resting_bytes = 0
        with mock.patch.object(m, "_headroom_bytes", lambda: grant_mib * (2 ** 20)), \
                mock.patch.object(m.cgroup, "read",
                                  lambda: {"current": None, "max": None, "available": None}):
            return m, m._solve_budget()

    def test_the_share_spends_half_the_grant(self):
        _m, budget = self._budget("")
        self.assertAlmostEqual(budget, 0.5 * 12288 / 6.0, places=3)

    def test_a_deployment_that_still_sets_the_removed_flag_is_told(self):
        # 🔴 An ENGINE_ARGS still carrying the flag has to be reported, or an operator goes
        # on setting a number that does nothing. Nothing else in either file holds this.
        with self.assertLogs("audio-stt-stream", level="WARNING") as caught:
            _stt("--batch auto --gpu-budget-fraction 5")
        self.assertTrue(any("--gpu-budget-fraction" in r.getMessage() for r in caught.records),
                        "a flag this engine no longer takes was dropped silently")

    def test_the_share_is_not_a_flag(self):
        # 🔴 A constant, not a knob: it is the only one that could put a batch ABOVE the
        # grant. Removing it is not silent -- an ENGINE_ARGS carrying it is reported.
        m, budget = self._budget("--batch auto --gpu-budget-fraction 5")
        self.assertEqual(m.BUDGET_FRACTION, 0.5)
        self.assertAlmostEqual(budget, 0.5 * 12288 / 6.0, places=3,
                               msg="a removed flag was still changing the budget")

    def test_no_counter_and_no_grant_are_told_apart_in_the_warning(self):
        # 🔴 Two situations, two fixes. No counter is a machine this engine cannot measure
        # and no operator can change; no grant is a declaration somebody can go and add.
        m = _stt()
        said = []
        with mock.patch.object(m, "_memory_reading", lambda: None), \
                mock.patch.object(m, "_p", said.append):
            self.assertIsNone(m._effective_budget())
        self.assertIn("no per-process memory counter", " ".join(said))
        self.assertNotIn("declares no GPU quota", " ".join(said))

        m = _stt()
        m._resting_bytes = 0
        said = []
        # No REQUIRED_GPU_MEMORY in this environment, so the quota really is absent --
        # which is the whole of the second case.
        with mock.patch.object(m, "_held_bytes", lambda: 0), \
                mock.patch.object(m, "_p", said.append):
            self.assertIsNone(m._effective_budget())
        self.assertIn("declares no GPU quota", " ".join(said))
        self.assertNotIn("no per-process memory counter", " ".join(said))

    def test_the_health_line_quotes_the_reading_its_own_number_came_from(self):
        # 🔴 Read once. A second `_headroom_bytes()` in the log line differs only on a real
        # machine, where the card moves between the two readings.
        m = _stt()
        m._resting_bytes = 0
        readings = [12288 * (2 ** 20), 1024 * (2 ** 20), 64 * (2 ** 20)]
        said = []
        with mock.patch.object(m, "_headroom_bytes", lambda: readings.pop(0)), \
                mock.patch.object(m.cgroup, "read",
                                  lambda: {"current": None, "max": None, "available": None}), \
                mock.patch.object(m, "_p", said.append):
            budget = m._effective_budget()
        self.assertIn("12288 MiB grant", " ".join(said),
                      "the line quoted a later reading of the card")
        self.assertAlmostEqual(12288 / 6.00 * 0.5, budget, places=3)

    def test_the_health_line_can_be_multiplied_back_out(self):
        # 🔴 The line has to be recomputable: 12288 MiB at 6.00 MiB a padded second is 2048,
        # not the 1024 printed. The missing term was the share, the only factor left out.
        m = _stt()
        m._resting_bytes = 0
        said = []
        with mock.patch.object(m, "_headroom_bytes", lambda: 12288 * (2 ** 20)), \
                mock.patch.object(m.cgroup, "read",
                                  lambda: {"current": None, "max": None, "available": None}), \
                mock.patch.object(m, "_p", said.append):
            budget = m._effective_budget()
        line = " ".join(said)
        self.assertIn("12288 MiB grant", line)
        self.assertIn("6.00 MiB a padded second", line)
        self.assertIn("spending 0.5 of it", line)
        self.assertAlmostEqual(12288 / 6.00 * 0.5, budget, places=3)

class HostAccountTest(unittest.TestCase):
    """The account that kills, which the GPU budget says nothing about."""

    # 🔴 mock.patch.object, not assignment. `cgroup` is a module and reloading the engine
    # does not replace it, so a plain assignment leaks into every later test in the file.
    def _module(self, limit, current=0, args=""):
        m = _stt(args)
        m._resting_bytes = 0
        patch = mock.patch.object(
            m.cgroup, "read",
            lambda: {"current": current, "max": limit, "available": None})
        self.addCleanup(patch.stop)
        patch.start()
        return m

    def _gpu_is_not_the_bound(self, m):
        patch = mock.patch.object(m, "_headroom_bytes", lambda: 100 * (2 ** 30))
        self.addCleanup(patch.stop)
        patch.start()

    def test_a_container_limit_bounds_the_batch_too(self):
        m = self._module(limit=2 * (2 ** 30))
        self._gpu_is_not_the_bound(m)
        # Half of 2 GiB at the floor times the overshoot. 🔴 Without the overshoot a budget
        # solved from the bare floor spends the whole container at the smallest rate seen.
        self.assertAlmostEqual(m._host_budget(),
                               0.5 * 2 * (2 ** 30) / (115200.0 * 8), places=3)

    def test_a_container_sitting_at_its_limit_forces_one_span_a_call(self):
        # 🔴 The zero this module exists for: with `host` instead of `host is not None`, a
        # container AT its limit is skipped as unread and OOMKilled -- 137, uncatchable.
        m = self._module(limit=2 * (2 ** 30), current=2 * (2 ** 30))
        self._gpu_is_not_the_bound(m)
        self.assertEqual(m._host_budget(), 0.0)
        groups, how = m._plan_groups([(0, "a", 30.0), (1, "b", 30.0), (2, "c", 30.0)])
        self.assertEqual(how, "one span a call")
        self.assertEqual([len(g) for g in groups], [1, 1, 1])

    def test_a_container_whose_limit_cannot_be_read_does_not_force_anything(self):
        # The other side, so the test above cannot pass by making every unreadable account
        # behave like a full one: nothing readable means the host has no opinion.
        m = _stt()
        m._resting_bytes = 0
        with mock.patch.object(m.cgroup, "read",
                               lambda: {"current": None, "max": None, "available": None}), \
                mock.patch.object(m, "_headroom_bytes", lambda: 100 * (2 ** 30)):
            self.assertIsNone(m._host_budget())
            _groups, how = m._plan_groups([(0, "a", 30.0), (1, "b", 30.0)])
        self.assertNotEqual(how, "one span a call")

    def test_a_reading_exactly_at_the_floor_is_not_a_measurement(self):
        # 🔴 The boundary: `>` is not `>=`. A reading exactly on the floor observed no call,
        # and treating it as measured drops the overshoot where the bound binds.
        m = _stt()
        floor = m.HOST_FLOOR_BYTES_A_PADDED_SECOND
        with mock.patch.object(m.cgroup, "read", lambda: {"current": int(floor * 60)}):
            self.assertIsNone(m._measure_host({"current": 0}, 60.0))
        with mock.patch.object(m.cgroup, "read", lambda: {"current": int(floor * 60) + 60}):
            self.assertIsNotNone(m._measure_host({"current": 0}, 60.0))

    def test_a_measured_cost_replaces_the_floor_and_its_overshoot(self):
        m = self._module(limit=2 * (2 ** 30))
        self._gpu_is_not_the_bound(m)
        m._host_bytes_a_padded_second = 400000.0
        self.assertAlmostEqual(m._host_budget(),
                               0.5 * 2 * (2 ** 30) / 400000.0, places=3)

    def test_a_reading_below_what_the_representation_costs_is_not_a_measurement(self):
        # 🔴 None, not the floor. The overshoot applies only where there is no measurement,
        # so answering "the floor" makes a barely-moving counter look eight times looser.
        m = _stt()
        with mock.patch.object(m.cgroup, "read", lambda: {"current": 1000}):
            self.assertIsNone(m._measure_host({"current": 0}, 60.0))

    def test_a_reading_above_the_floor_is_taken_as_measured(self):
        m = _stt()
        spent = 60.0 * m.HOST_FLOOR_BYTES_A_PADDED_SECOND * 3
        with mock.patch.object(m.cgroup, "read", lambda: {"current": spent}):
            self.assertAlmostEqual(m._measure_host({"current": 0}, 60.0),
                                   m.HOST_FLOOR_BYTES_A_PADDED_SECOND * 3)

    def test_the_delta_is_taken_on_what_the_bound_spends(self):
        # ⚠️ Same quantity as cgroup.headroom: current minus reclaimable. A raw
        # memory.current delta charges page cache against an account that does not count it.
        m = _stt()
        big = 60.0 * m.HOST_FLOOR_BYTES_A_PADDED_SECOND * 5
        with mock.patch.object(m.cgroup, "read",
                               lambda: {"current": big, "reclaimable": big}):
            self.assertIsNone(m._measure_host({"current": 0, "reclaimable": 0}, 60.0))

    def test_an_unreadable_container_measures_nothing_rather_than_zero(self):
        m = _stt()
        with mock.patch.object(m.cgroup, "read", lambda: {"current": None}):
            self.assertIsNone(m._measure_host({"current": 0}, 60.0))

    def test_the_smaller_of_the_two_accounts_wins_and_the_report_says_which(self):
        m = self._module(limit=64 * (2 ** 20))          # tiny container, huge GPU grant
        self._gpu_is_not_the_bound(m)
        groups, how = m._plan_groups([(0, [], 30.0)] * 4)
        self.assertIn("container's memory", how)
        self.assertTrue(groups)

    def test_an_unreadable_cgroup_is_no_host_bound_rather_than_no_room(self):
        # 🔴 The third answer again. A machine with no readable cgroup must not collapse to
        # one span a call -- that would be a bound nobody set, applied everywhere.
        m = self._module(limit=None)
        self._gpu_is_not_the_bound(m)
        self.assertIsNone(m._host_budget())
        _groups, how = m._plan_groups([(0, [], 30.0)] * 4)
        self.assertNotIn("container's memory", how)

    def test_a_count_set_too_high_can_no_longer_build_the_call_that_killed_us(self):
        # 🔴 The shape that killed the engine on the previous runtime: a span count set too
        # high, which knows nothing about what a call costs. It cannot be built any more --
        # a count is a ceiling, and the harness's machine offers nothing to size from, so
        # 200 spans of 30 seconds is 200 calls rather than one of 6,000 padded seconds.
        m = self._module(limit=8 * (2 ** 20), args="--batch-max-spans 200")
        groups, how = m._plan_groups([(i, [], 30.0) for i in range(200)])
        self.assertEqual(len(groups), 200, "200 spans went out in fewer calls than spans")
        self.assertEqual(how, "one span a call")
        self.assertEqual(sorted(i for g in groups for i, _c, _s in g), list(range(200)))


class PlanGroupsTest(unittest.TestCase):
    """How a list of spans becomes calls, and what the count does on top."""

    SPANS = [(0, [], 5.0), (1, [], 300.0), (2, [], 30.0), (3, [], 30.0), (4, [], 5.0)]

    def _plan(self, engine_args, budget=None):
        m = _stt(engine_args)
        if budget is not None:
            m._headroom_bytes = lambda: budget * m._bytes_a_padded_second() / m.BUDGET_FRACTION
        # 🔴 The container account is stubbed, not left to the test host. _plan_groups
        # consults it on every path, so a 1 GiB limit would split the 600 second groups.
        patch = mock.patch.object(m.cgroup, "read",
                                  lambda: {"current": None, "max": None, "available": None})
        self.addCleanup(patch.stop)
        patch.start()
        groups, how = m._plan_groups(self.SPANS)
        return m, [[i for i, _c, _s in g] for g in groups], how

    def test_a_count_on_its_own_cannot_build_a_call(self):
        # 🔴 The semantic a count carries now. It used to select a mode that grouped by
        # arrival with no memory bound at all; it is a ceiling, and a ceiling cannot create
        # a batch on a machine that cannot be measured -- the harness has no grant here.
        _m, groups, how = self._plan("--batch-max-spans 2")
        self.assertEqual(groups, [[0], [1], [2], [3], [4]])
        self.assertEqual(how, "one span a call")

    def test_a_count_of_one_does_not_group_at_all(self):
        m = _stt("--batch-max-spans 1")
        self.assertFalse(m.GROUPING)

    def test_an_unmeasured_machine_says_its_groups_are_not_evidence(self):
        # 🔴 A fourth state, borrowed from the alignment line: "one span because the opening
        # guess was small" and "one span because the card is full" must not read alike.
        m = _stt()
        m._headroom_bytes = lambda: 12 * (2 ** 30)
        self.assertEqual(m._scale_seen, 0)
        _groups, how = m._plan_groups(self.SPANS)
        self.assertIn("has been measured", how)
        m._observe(60.0, 60.0 * m.OPENING_BYTES_A_PADDED_SECOND)
        _groups, how = m._plan_groups(self.SPANS)
        self.assertNotIn("has been measured", how)

    def test_a_seconds_budget_sorts_and_packs_by_cost(self):
        # 🔴 Two calls either way, so the 300 travels ALONE: dragging a 30 s span with it
        # costs 600 + 90 where letting it go alone costs 300 + 120. Filling the group to
        # capacity bought nothing here, which is the case the exact packer exists for.
        _m, groups, how = self._plan("", budget=600.0)
        self.assertEqual(groups, [[1], [2, 3, 0, 4]])
        self.assertIn("600 padded seconds", how)

    def test_a_ceiling_given_with_a_seconds_budget_still_applies(self):
        _m, groups, how = self._plan("--batch-max-spans 2", budget=600.0)
        self.assertTrue(all(len(g) <= 2 for g in groups), groups)
        self.assertIn("at most 2 spans", how)

    def test_no_budget_is_one_span_a_call_and_says_so(self):
        # 🔴 With the container readable, which is the case that used to slip through: a
        # host budget must not substitute for a GPU budget that does not exist.
        m = _stt()
        m._headroom_bytes = lambda: None
        cg = mock.patch.object(m.cgroup, "read",
                               lambda: {"current": 0, "max": 8 * (2 ** 30),
                                        "available": None, "reclaimable": 0})
        self.addCleanup(cg.stop)
        cg.start()
        groups, how = m._plan_groups(self.SPANS)
        self.assertEqual([len(g) for g in groups], [1, 1, 1, 1, 1])
        self.assertEqual(how, "one span a call")

    def test_every_span_survives_every_configuration(self):
        # ⚠️ Three configurations, which is all there are: no ceiling, a ceiling, and the
        # ceiling of one that turns grouping off. What stood here listed four "modes" of
        # which three were the same one under flags this engine no longer takes.
        for args in ("", "--batch-max-spans 2", "--batch-max-spans 1"):
            m = _stt(args)
            m._headroom_bytes = lambda: 8 * (2 ** 30)
            groups, _how = m._plan_groups(self.SPANS)
            self.assertEqual(sorted(i for g in groups for i, _c, _s in g),
                             [0, 1, 2, 3, 4], "%s lost or duplicated a span" % args)


class UnsplittableMakesRoomTest(unittest.TestCase):
    """The one group nothing can shrink, and the only lever left before it goes out."""

    def _plan(self, m, seconds, budget):
        spans = [(i, "clip", s) for i, s in enumerate(seconds)]
        dropped = []
        with mock.patch.object(m, "_peak_bytes", lambda: None), \
                mock.patch.object(m, "_effective_budget", lambda: budget), \
                mock.patch.object(m, "_drop_cache", lambda: dropped.append(True)):
            groups, how = m._plan_groups(spans)
        return groups, how, dropped

    def test_a_single_span_over_the_budget_gets_the_cache_handed_back_first(self):
        # 🔴 596 padded seconds against a budget of 33 is the shape that ended in exit 137.
        # Nothing can split it, so the only thing left is to stop holding what is not in use.
        m = _stt()
        groups, how, dropped = self._plan(m, [595.0], budget=33.0)
        self.assertEqual([len(g) for g in groups], [1])
        self.assertEqual(len(dropped), 1, "the cache was not handed back")
        self.assertIn("cache was handed back", how)

    def test_a_single_span_inside_the_budget_is_left_alone(self):
        # The call fits. Dropping here would pay empty_cache() on the ordinary path, which is
        # what keeps this out of every request.
        m = _stt()
        _groups, how, dropped = self._plan(m, [30.0], budget=600.0)
        self.assertEqual(dropped, [])
        self.assertNotIn("cache was handed back", how)

    def test_a_group_that_can_still_be_split_is_left_alone(self):
        """🔴 Twenty spans over budget are not this case: the planner already split them, and
        every group it produced fits. Dropping here would fire on ordinary traffic."""
        m = _stt()
        groups, _how, dropped = self._plan(m, [30.0] * 20, budget=200.0)
        self.assertTrue(all(len(g) > 1 for g in groups), groups)
        self.assertEqual(dropped, [])
class LongSpanTest(unittest.TestCase):
    """The one place the cost model is known to be wrong, and has to say so."""

    def _plan(self, seconds):
        # ⚠️ Only the long-span line. The health line fires here too, once, and counting it
        # would make this test about how many lines a first request prints.
        m = _stt()
        said = []
        with mock.patch.object(m, "_p", said.append):
            m._plan_groups([(0, [], seconds)])
            m._plan_groups([(0, [], seconds)])
        return m, [x for x in said if "in one piece" in x]

    def test_a_span_past_what_the_model_takes_in_one_piece_is_named(self):
        # 🔴 The library splits it and pads the pieces together, so the call costs nearly
        # twice what a model that counts it as one clip predicts. Not corrected, said.
        _m, said = self._plan(600.0)
        self.assertEqual(len(said), 1, "said once, not once a request")
        self.assertIn("UNDER-counts", said[0])
        self.assertIn("600s span", said[0])

    def test_a_span_inside_it_says_nothing(self):
        _m, said = self._plan(300.0)
        self.assertEqual(said, [])

    def test_the_boundary_is_what_the_model_takes_in_one_piece(self):
        # Exactly the limit is still one piece -- the library splits what is PAST it. A
        # warning here would send someone looking for a split that did not happen.
        m = _stt()
        at_the_limit = [x for x in self._only(m, float(m.OFFLINE_MAX_INPUT_SEC))]
        self.assertEqual(at_the_limit, [])
        past_it = [x for x in self._only(m, m.OFFLINE_MAX_INPUT_SEC + 0.5)]
        self.assertEqual(len(past_it), 1)

    def _only(self, m, seconds):
        """The long-span line alone; the health line fires here too, once."""
        said = []
        with mock.patch.object(m, "_p", said.append):
            m._plan_groups([(0, [], seconds)])
        return [x for x in said if "in one piece" in x]


class RefusalTest(unittest.TestCase):
    """Which failures move the budget. 🔴 Only the card's, and it is load-bearing: any
    other failure says nothing about size, and treating one as a refusal shrinks the
    budget for the rest of a request over, say, one malformed span."""

    def test_the_card_refusing_is_recognised_by_class_and_by_message(self):
        m = _stt()

        class OutOfMemoryError(RuntimeError):
            pass

        self.assertTrue(m._is_oom(OutOfMemoryError("no room")))
        self.assertTrue(m._is_oom(RuntimeError("CUDA out of memory. Tried to allocate...")))

    def test_anything_else_failing_is_not_a_refusal(self):
        m = _stt()
        self.assertFalse(m._is_oom(RuntimeError("engine returned 2 results for 3 spans")))
        self.assertFalse(m._is_oom(ValueError("could not convert string to float")))

    def test_the_ceiling_lifts_after_the_recovery_window(self):
        # 🔴 One refusal caused by a neighbour's minute must not shrink this process for the
        # life of a model server. A wrong retry costs ONE refused call, already handled.
        m = _stt()
        m._resting_bytes = 0
        with mock.patch.object(m.time, "monotonic", lambda: 1000.0):
            m._note_refusal(400.0)
            self.assertEqual(m._refusal_ceiling(), 200.0)
        with mock.patch.object(m.time, "monotonic",
                               lambda: 1000.0 + m.RECOVERY_SECONDS - 0.1):
            self.assertEqual(m._refusal_ceiling(), 200.0, "it lifted early")
        with mock.patch.object(m.time, "monotonic", lambda: 1000.0 + m.RECOVERY_SECONDS):
            self.assertIsNone(m._refusal_ceiling(), "it did not lift")

    def test_lifting_forgets_the_refusal_rather_than_ignoring_it(self):
        # 🔴 Clearing, not ignoring: an ignored refusal stays in the minimum below, so a
        # tiny transient an hour ago bounds a fresh refusal twenty times its size.
        m = _stt()
        with mock.patch.object(m.time, "monotonic", lambda: 1000.0):
            m._note_refusal(20.0)
        with mock.patch.object(m.time, "monotonic", lambda: 1000.0 + m.RECOVERY_SECONDS):
            self.assertIsNone(m._refusal_ceiling())
            m._note_refusal(400.0)
            self.assertEqual(m._refusal_ceiling(), 200.0,
                             "a refusal that had aged out was still setting the ceiling")

    def test_a_second_refusal_inside_the_window_still_takes_the_smaller(self):
        m = _stt()
        with mock.patch.object(m.time, "monotonic", lambda: 1000.0):
            m._note_refusal(400.0)
        with mock.patch.object(m.time, "monotonic", lambda: 1030.0):
            m._note_refusal(900.0)
            self.assertEqual(m._refusal_ceiling(), 200.0,
                             "a later, larger refusal raised the ceiling")

    def test_the_budget_goes_back_to_the_arithmetic_once_the_ceiling_lifts(self):
        m = _stt()
        m._resting_bytes = 0
        with mock.patch.object(m, "_headroom_bytes", lambda: 12288 * (2 ** 20)), \
                mock.patch.object(m.cgroup, "read",
                                  lambda: {"current": None, "max": None, "available": None}):
            with mock.patch.object(m.time, "monotonic", lambda: 1000.0):
                m._note_refusal(400.0)
                self.assertEqual(m._solve_budget(), 200.0)
            with mock.patch.object(m.time, "monotonic",
                                   lambda: 1000.0 + m.RECOVERY_SECONDS):
                self.assertAlmostEqual(m._solve_budget(), 0.5 * 12288 / 6.0, places=3)

    def test_a_refusal_is_remembered_at_its_smallest(self):
        # 🔴 The smallest is not last, or "keep the minimum" and "keep the latest" agree.
        # "Latest" raises a ceiling of 100 to 300 -- a bigger batch right after a refusal.
        m = _stt()
        m._note_refusal(200.0)
        m._note_refusal(900.0)
        self.assertEqual(m._refused_above, 200.0, "a later, larger refusal raised the ceiling")

    def test_a_refusal_of_nothing_is_not_a_ceiling_of_nothing(self):
        m = _stt()
        m._note_refusal(0.0)
        self.assertIsNone(m._refused_above)


class SmallerPlanTest(unittest.TestCase):
    """The structural guard that makes the retry loop terminate.

    🔴 Deliberately not the same statement as "the budget came down": a loop whose
    termination rests on a comparison made elsewhere spins the day that one changes.
    """

    def _module(self, budget):
        m = _stt()
        m._headroom_bytes = lambda: budget * m._bytes_a_padded_second() / m.BUDGET_FRACTION
        return m

    def _spans(self, n, seconds=1.0):
        return [(i, [], seconds) for i in range(n)]

    def test_a_plan_that_is_smaller_comes_back(self):
        m = self._module(4.0)          # four 1 s spans a call
        refused = self._spans(8)
        plan, _how = m._smaller_plan([], refused)
        self.assertLessEqual(max(len(g) for g in plan), 4)
        self.assertEqual(sorted(i for g in plan for i, _c, _s in g), list(range(8)))

    def test_the_re_plan_comes_back_longest_first(self):
        # 🔴 The caller pops from the END, so the reverse is what dispatches the plan's
        # first group -- the longest-headed one, most likely to be refused again -- first.
        m = self._module(240.0)
        refused = [(0, [], 120.0), (1, [], 5.0), (2, [], 5.0), (3, [], 5.0)]
        plan, _how = m._smaller_plan([], refused)
        first_out = plan[-1]
        self.assertIn(120.0, [one[2] for one in first_out],
                      "the longest span was not in the group dispatched first")

    def test_a_plan_that_is_not_smaller_is_refused(self):
        # The budget still allows the group that just failed, so re-planning would hand
        # back the same plan. None says "halve instead", which always makes progress.
        m = self._module(100.0)
        self.assertIsNone(m._smaller_plan([], self._spans(8)))

    def test_it_carries_the_groups_still_queued_not_only_the_failed_one(self):
        m = self._module(4.0)
        queued = [self._spans(3)]
        refused = [(9, [], 1.0), (10, [], 1.0), (11, [], 1.0), (12, [], 1.0),
                   (13, [], 1.0), (14, [], 1.0)]
        plan, _how = m._smaller_plan(queued, refused)
        self.assertEqual(sorted(i for g in plan for i, _c, _s in g),
                         [0, 1, 2, 9, 10, 11, 12, 13, 14])

    def test_a_group_of_one_can_never_be_re_planned_smaller(self):
        # Where the recursion has to stop: one span is the smallest call there is, and the
        # handler's ordinary error path takes it from here.
        m = self._module(0.0001)
        self.assertIsNone(m._smaller_plan([], [(0, [], 1.0)]))


class CallGroupTest(unittest.TestCase):
    def _module(self, used):
        m = _stt()
        readings = [(0, 0), (0, used)]
        seq = list(readings)
        m._memory_reading = lambda: seq.pop(0) if len(seq) > 1 else seq[0]
        m._offline_transcribe_many = lambda clips, language=None, context="": [
            ("", "")
        ] * len(clips)
        return m

    def test_a_call_measures_what_its_padded_seconds_cost(self):
        # Three clips whose longest is 20 s cost 3 x 21, not the 35 they add up to and not the
        # 60 they would be without the floor every span carries whatever its length.
        m = self._module(used=63 * 6 * (2 ** 20))
        m._call_group([(0, [], 20.0), (1, [], 10.0), (2, [], 5.0)])
        self.assertAlmostEqual(m._scale, 1.0, places=6)
        self.assertEqual((m._scale_seen, m._calls_seen), (1, 1))

    def test_a_reading_that_did_not_rise_teaches_nothing_but_still_counts(self):
        m = self._module(used=0)
        m._call_group([(0, [], 20.0)])
        self.assertEqual((m._scale_seen, m._calls_seen), (0, 1))


class BatchingReportTest(unittest.TestCase):
    """The one line anything outside this process can read the sizing from."""

    @staticmethod
    def _plan(*groups):
        """[(seconds, ...)] -> the shape _plan_groups hands back."""
        at, out = 0, []
        for group in groups:
            rows = []
            for seconds in group:
                rows.append((at, "clip", float(seconds)))
                at += 1
            out.append(rows)
        return out

    def _line(self, m, planned, **kw):
        kw.setdefault("how", "auto")
        kw.setdefault("calls", len(planned))
        return m._batching_report(kw["how"], sum(len(g) for g in planned), planned,
                                  kw["calls"], kw.get("splits", 0), kw.get("refusals", 0),
                                  kw.get("replans", 0), kw.get("replanned_how"))

    def test_the_line_says_what_each_call_costs_not_only_how_many_there_are(self):
        # 🔴 Two plans of the same 21 spans in the same 2 calls can be five times apart in
        # cost. The count cannot tell them apart, and the count was all this line had.
        m = _stt()
        with mock.patch.object(m, "_peak_bytes", lambda: None):
            packed = self._line(m, self._plan([120.0] * 5, [5.0] * 16))
            by_count = self._line(m, self._plan([120.0] + [5.0] * 11, [5.0] * 9))
        self.assertIn("[5x120s 16x5s] = 701 padded seconds", packed)
        self.assertIn("[12x120s 9x5s] = 1506 padded seconds", by_count)
        self.assertIn("2 calls", packed)
        self.assertIn("2 calls", by_count)

    def test_a_re_plan_is_labelled_rather_than_blended_into_the_first_plan(self):
        # 🔴 The shape and the padded seconds describe the plan the SIZER chose. Quoting a
        # re-plan's budget beside the first plan's shape puts two plans in one sentence.
        m = _stt()
        with mock.patch.object(m, "_peak_bytes", lambda: None):
            line = self._line(m, self._plan([120.0] * 5, [5.0] * 16), how="639 padded seconds",
                              calls=7, refusals=1, replans=1,
                              replanned_how="300 padded seconds a call (and a note)")
        self.assertIn("(639 padded seconds) as [5x120s 16x5s] = 701 padded seconds", line)
        self.assertIn("re-planned at 300 padded seconds a call", line)
        self.assertNotIn("and a note", line)

    def test_no_counter_and_a_peak_of_nothing_are_different_words(self):
        # The module's discipline: no counter and a peak of zero have different causes and
        # different fixes, and "0 MiB" for the first cannot be argued back out.
        m = _stt()
        with mock.patch.object(m, "_peak_bytes", lambda: None):
            self.assertIn("peak since load unmeasured", self._line(m, self._plan([5.0])))
        with mock.patch.object(m, "_peak_bytes", lambda: 0):
            self.assertIn("peak since load 0 MiB", self._line(m, self._plan([5.0])))

    def test_the_peak_is_the_high_water_mark_and_is_reported_in_mib(self):
        m = _stt()
        with mock.patch.object(m, "_peak_bytes", lambda: 4619 * (2 ** 20)):
            self.assertIn("peak since load 4619 MiB", self._line(m, self._plan([5.0])))

    def test_an_empty_group_does_not_take_the_line_down(self):
        # _plan_shape divides nothing, but a plan that somehow carries an empty group must
        # not cost the report -- it is the only record the request leaves.
        m = _stt()
        with mock.patch.object(m, "_peak_bytes", lambda: None):
            self.assertIn("0x0s", self._line(m, [[]]))


class CalibrationTest(unittest.TestCase):
    """What warmup reports, in each of the three states it can end in."""

    def _run(self, readings, transcribe=None):
        """readings: successive (allocated, peak) tuples _memory_reading() will answer.

        Two of them, one either side of the calibration call. The reset precedes both.
        """
        m = _stt()
        said, order, seq = [], [], list(readings)

        def reading():
            return seq.pop(0) if len(seq) > 1 else seq[0]

        with mock.patch.object(m, "_p", said.append), \
                mock.patch.object(m, "_memory_reading", reading), \
                mock.patch.object(m, "_reset_peak", lambda: order.append("reset")), \
                mock.patch.object(m, "_offline_transcribe_many",
                                  transcribe or (lambda clips: order.append("call") or
                                                 [("", "")] * len(clips))), \
                mock.patch.object(m, "_offline_transcribe", lambda a, language=None: ""), \
                mock.patch.object(m, "_drop_cache", lambda: order.append("drop")):
            m._warmup()
        return m, said, order

    def test_the_peak_is_reset_before_the_calibration_call_not_after(self):
        # 🔴 Trap one. Loading the model sets a peak of its own, and a call that does not
        # rise above it measures zero -- while the line below still says "calibrated".
        _m, _said, order = self._run([(100, 100), (100, 700)])
        # The first two steps only; what follows the call is pinned below, so one change
        # does not break two tests for one reason.
        self.assertEqual(order[:2], ["reset", "call"])

    def test_measuring_nothing_and_having_no_counter_are_two_different_lines(self):
        # 🔴 Trap two: a log that prints a number either way cannot be read backwards.
        _m, flat, _o = self._run([(100, 100), (100, 100)])
        _m2, none, _o2 = self._run([None])
        self.assertIn("took no measurement", " ".join(flat))
        self.assertIn("no per-process memory counter", " ".join(none))
        self.assertNotIn("took no measurement", " ".join(none))

    def test_the_calibration_shape_is_the_length_it_says_it_is(self):
        # The correction is measured cost over PREDICTED cost, and the prediction is
        # computed from these seconds. A tone of the wrong length quietly scales it.
        m = _stt()
        self.assertEqual(len(m._tone(2.0)), 32000)

    def test_the_path_with_no_counter_warms_on_the_cheapest_shape(self):
        # 🔴 This is also the CPU path, where the calibration shape is minutes of warmup
        # for a measurement that cannot be taken anyway.
        m = _stt()
        lengths = []
        with mock.patch.object(m, "_p", lambda _x: None), \
                mock.patch.object(m, "_memory_reading", lambda: None), \
                mock.patch.object(m, "_offline_transcribe",
                                  lambda a, language=None: lengths.append(len(a))):
            m._warmup()
        self.assertEqual(lengths, [16000])

    def test_a_small_but_real_reading_is_still_a_measurement(self):
        # 🔴 Any rise above the peak is a sample. No floor on purpose: a floor would be a
        # number nobody measured, and the band above already says when the ratios stop.
        m, said, _o = self._run([(0, 0), (0, 100)])
        self.assertEqual(m._calls_seen, 1)
        self.assertNotIn("took no measurement", " ".join(said))

    def test_a_measurement_reaches_the_correction(self):
        m, said, _o = self._run([(0, 0), (0, 2 * 60 * 6 * (2 ** 20))])
        self.assertAlmostEqual(m._scale, 2.0)
        line = " ".join(said)
        self.assertIn("calibrated in", line)
        self.assertIn("padded seconds", line)

    def test_a_failed_warmup_says_so_and_leaves_the_correction_alone(self):
        def boom(clips):
            raise RuntimeError("card is full")

        m, said, order = self._run([(0, 0), (0, 999)], transcribe=boom)
        self.assertAlmostEqual(m._scale, 1.0)
        self.assertIn("warmup transcription failed", " ".join(said))
        # 🔴 Both happen BEFORE the call, and that ordering is the protection: a reset
        # inside the guarded block leaves `_used` answering zero for the process's life.
        self.assertEqual(order, ["reset", "drop"],
                         "a failed calibration kept the blocks it had allocated")
        self.assertEqual(m._resting_bytes, 0)

    def test_the_peak_is_reset_before_anything_that_can_raise(self):
        # 🔴 Structural, not argued: the reset is the first statement, so every reader in
        # _warmup can blow up and the peak has still been reset.
        m = _stt()
        order, said = [], []
        with mock.patch.object(m, "_reset_peak", lambda: order.append("reset")), \
                mock.patch.object(m.cgroup, "read", side_effect=OSError("no cgroup")), \
                mock.patch.object(m, "_p", said.append):
            m._warmup()
        self.assertEqual(order, ["reset"])
        # 🔴 And it does not escape. A reader that blew up propagated out of _warmup,
        # `ready` was never set, and every later request answered 503 for the process's life.
        self.assertIn("could not finish measuring", " ".join(said))
        self.assertIn("serviceable", " ".join(said))

    def test_the_calibration_hands_its_blocks_back_after_it_has_measured(self):
        # 🔴 A caching allocator keeps the calibration batch: one batched request took this
        # engine 8290 -> 14460 MiB. ⚠️ Pins the POSITION only; the return needs a card.
        _m, _said, order = self._run([(1000, 1000), (1000, 7000)])
        self.assertEqual(order, ["reset", "call", "drop"])

    def test_a_calibration_that_measured_nothing_still_releases(self):
        # 🔴 `used <= 0` means the COUNTER did not see it -- another device, a virtualized
        # view -- not that nothing ran. The allocator is holding blocks in every one.
        _m, said, order = self._run([(100, 700), (100, 700)])
        self.assertEqual(order, ["reset", "call", "drop"])
        self.assertIn("took no measurement", " ".join(said))

    def test_what_the_process_holds_at_rest_is_read_before_the_call(self):
        m, _said, _o = self._run([(1234, 1234), (1234, 5000)])
        self.assertEqual(m._resting_bytes, 1234)


if __name__ == "__main__":
    unittest.main()

class RecoveryBackoffTest(unittest.TestCase):
    """What makes a wrong cost model stop costing a call a minute, forever.

    🔴 A refusal never reaches `_observe` -- a call that died allocated nothing to measure --
    so the model cannot learn from it. The ceiling lifting on a fixed timer therefore restores
    the same wrong budget, and a model that is wrong about THIS machine refuses again on the
    next request, every minute, for the life of the process. The wait is the only thing that
    can tell the two causes apart: a neighbour's minute is gone by the first retry.
    """

    def _refuse(self, m, now, size):
        with mock.patch.object(m.time, "monotonic", lambda: now):
            m._note_refusal(size)

    def _ceiling(self, m, now):
        with mock.patch.object(m.time, "monotonic", lambda: now):
            return m._refusal_ceiling()

    def test_one_refusal_still_lifts_after_a_minute(self):
        m = _stt()
        self._refuse(m, 1000.0, 400.0)
        self.assertEqual(self._ceiling(m, 1059.0), 200.0)
        self.assertIsNone(self._ceiling(m, 1061.0))

    def test_a_refusal_that_survives_its_own_recovery_doubles_the_wait(self):
        # 🔴 The discrimination: this is the second ceiling, not the second refusal. A transient
        # cannot produce it, because the first retry already went through.
        m = _stt()
        self._refuse(m, 1000.0, 400.0)
        self.assertIsNone(self._ceiling(m, 1061.0))
        self._refuse(m, 1100.0, 400.0)
        self.assertIsNotNone(self._ceiling(m, 1100.0 + 90.0),
                             "the second ceiling lifted on the first ceiling's timer")
        self.assertIsNone(self._ceiling(m, 1100.0 + 121.0))

    def test_re_planning_inside_one_request_is_one_piece_of_evidence(self):
        # 🔴 Refusals, not ceilings, would count a single wrong budget three times and put the
        # wait at eight minutes for what is one request's worth of evidence.
        m = _stt()
        self._refuse(m, 1000.0, 400.0)
        self._refuse(m, 1001.0, 200.0)
        self._refuse(m, 1002.0, 100.0)
        self.assertEqual(m._refusal_streak, 1)
        self.assertIsNone(self._ceiling(m, 1002.0 + 61.0))

    def test_the_streak_clears_along_the_path_it_actually_takes(self):
        # 🔴 Driven end to end rather than by calling the clearer with a state the engine never
        # reaches. The first version of this test set the ceiling and called _note_fit(400)
        # straight after -- which cannot happen, because while the ceiling stands the budget is
        # HALF the refused size, so no call can reach it. The clear has to survive the ceiling
        # expiring, or the backoff only ever grows and one transient shrinks the process for good.
        m = _stt()
        m._resting_bytes = 0
        m._headroom_bytes = lambda: int(800.0 * 6.0 * (2 ** 20) / 0.5)
        with mock.patch.object(m.cgroup, "read",
                               lambda: {"current": None, "max": None, "available": None}):
            self._refuse(m, 1000.0, 400.0)
            with mock.patch.object(m.time, "monotonic", lambda: 1000.0):
                under = m._effective_budget()
            self.assertLess(under, 400.0, "a call under the ceiling could reach the refused size")
            with mock.patch.object(m.time, "monotonic", lambda: 1061.0):
                lifted = m._effective_budget()
            self.assertGreater(lifted, 400.0)
            self.assertIsNone(m._refused_above, "the ceiling did not clear")
            m._note_fit(lifted)
        self.assertEqual(m._refusal_streak, 0, "nothing can ever clear the streak")

    def test_a_smaller_call_fitting_proves_nothing(self):
        # 🔴 Smaller calls fitting is what the ceiling was FOR. Clearing on them would reset the
        # streak on the very re-plan the refusal caused, and the backoff would never start.
        m = _stt()
        self._refuse(m, 1000.0, 400.0)
        m._note_fit(50.0)
        self.assertEqual(m._refusal_streak, 1)

    def test_the_size_to_beat_outlives_the_ceiling(self):
        m = _stt()
        self._refuse(m, 1000.0, 400.0)
        self.assertIsNone(self._ceiling(m, 1061.0), "the ceiling should have lifted")
        self.assertEqual(m._streak_target, 400.0, "the size to beat went with the ceiling")

    def test_the_line_quotes_the_wait_that_is_actually_in_force(self):
        """🔴 The backoff exists to be SEEN, and the line was printing the constant.

        A second refusal doubles how long the ceiling stands. The report said "lifts 60s"
        either way, so an operator reading it came back at a minute to a process that would
        not be sized from the arithmetic again for two. The two smoke checks that pin this
        sentence both run at a streak of one, where the constant and the real wait agree --
        so nothing anywhere caught it.
        """
        m = _stt()
        m._refusal_streak, m._refused_above = 2, 400.0
        m._refused_at = m.time.monotonic()
        with mock.patch.object(m, "_peak_bytes", lambda: None), \
                mock.patch.object(m, "_budget_now", lambda: 200.0):
            planned = [[(0, "clip", 30.0)]]
            line = m._batching_report("auto", 1, planned, 1, 0, 1, 1, None)
        self.assertIn("lifts %ds after the refusal" % (m.RECOVERY_SECONDS * 2), line)
        self.assertNotIn("lifts %ds after the refusal" % m.RECOVERY_SECONDS, line)

    def test_the_wait_is_capped(self):
        m = _stt()
        m._refusal_streak = 99
        self.assertEqual(m._recovery_seconds(),
                         m.RECOVERY_SECONDS * (2 ** m.RECOVERY_DOUBLINGS_MAX))


class BothAccountsInOnePlaceTest(unittest.TestCase):
    """The size the log promises and the size that gets planned are the same number.

    🔴 They were not: `_budget_now` solved the GPU side only, while `_plan_groups` took the
    smaller of it and the container's. The line exists so the sizing can be audited from
    outside the process, and where the container bound it the line overstated by 3.5x --
    in the same sentence that named the container as the thing holding it down.
    """

    def _module(self, gpu_seconds, host_limit):
        m = _stt()
        m._resting_bytes = 0
        m._headroom_bytes = lambda: int(gpu_seconds * 6.0 * (2 ** 20) / 0.5)
        patch = mock.patch.object(m.cgroup, "read",
                                  lambda: {"current": 0, "max": host_limit,
                                           "available": None, "reclaimable": 0})
        patch.start()
        self.addCleanup(patch.stop)
        return m

    def test_the_budget_quoted_is_the_budget_planned_against(self):
        m = self._module(gpu_seconds=1024.0, host_limit=512 * (2 ** 20))
        said = m._budget_now()
        groups, _how = m._plan_groups([(i, None, 30.0) for i in range(60)])
        self.assertLessEqual(max(len(g) for g in groups) * 30.0, said,
                             "a group was planned bigger than the line said it would be")
        self.assertLess(said, m._solve_budget(), "the container did not lower anything")

    def test_an_unreadable_container_still_leaves_the_gpu_budget_alone(self):
        m = _stt()
        m._resting_bytes = 0
        m._headroom_bytes = lambda: int(600.0 * 6.0 * (2 ** 20) / 0.5)
        with mock.patch.object(m.cgroup, "read",
                               lambda: {"current": None, "max": None, "available": None}):
            self.assertAlmostEqual(m._budget_now(), 600.0, places=3)

    def test_a_container_at_its_limit_forces_one_span_a_call_through_this_reader(self):
        # 🔴 Zero, not None: the host account may lower a budget to nothing, and that is an
        # instruction. It may never stand in for a GPU grant that could not be read at all.
        m = self._module(gpu_seconds=1024.0, host_limit=0)
        self.assertEqual(m._budget_now(), 0.0)

    def test_the_host_cannot_invent_a_budget_where_the_gpu_has_no_authority(self):
        m = _stt()
        m._resting_bytes = 0
        m._headroom_bytes = lambda: None
        with mock.patch.object(m.cgroup, "read",
                               lambda: {"current": 0, "max": 8 * (2 ** 30),
                                        "available": None, "reclaimable": 0}):
            self.assertIsNone(m._budget_now())

    def test_ov_unified_memory_lets_the_host_stand_in(self):
        env = {"AUDIO_BASE": "ov", "MODEL_SUPPORTS": "stt,stt_stream", "ENGINE_ARGS": ""}
        with mock.patch.dict(os.environ, env, clear=False):
            m = importlib.reload(importlib.import_module("wrapper.caps.stt_stream"))
            self.addCleanup(lambda: _stt())
            m._resting_bytes = 0
            m._headroom_bytes = lambda: None
            with mock.patch.object(m.cgroup, "read",
                                   lambda: {"current": 0, "max": 8 * (2 ** 30),
                                            "available": None, "reclaimable": 0}):
                said = m._budget_now()
        self.assertIsNotNone(said)
        self.assertGreater(said, 0)


class ReplanAfterAMeasurementTest(unittest.TestCase):
    """A call that proves the factory rate is high must shrink the groups still waiting.

    The cgroup kills the next group. That kill is not an exception, so the
    outstanding plan has to move before the group runs.
    """

    def _module(self, budget_seconds):
        m = _stt()
        m._resting_bytes = 0
        m._headroom_bytes = lambda: int(budget_seconds * 6.0 * (2 ** 20) / 0.5)
        patch = mock.patch.object(m.cgroup, "read",
                                  lambda: {"current": None, "max": None, "available": None})
        patch.start()
        self.addCleanup(patch.stop)
        return m

    def test_a_group_over_the_new_budget_is_repacked(self):
        m = self._module(200.0)
        todo = [[(i, None, 4.0) for i in range(40)]]
        self.assertGreater(m.grouping.padded_seconds([4.0] * 40), 0)
        m._headroom_bytes = lambda: int(40.0 * 6.0 * (2 ** 20) / 0.5)
        shrunk = m._replan_over_budget(todo)
        self.assertIsNotNone(shrunk)
        groups, _how = shrunk
        budget = m._budget_now()
        for group in groups:
            self.assertLessEqual(m.grouping.padded_seconds([s for _i, _c, s in group]), budget)

    def test_an_unmeasured_ov_budget_is_a_quarter_of_the_factory_one(self):
        env = {"AUDIO_BASE": "ov", "OLARES_GPU_MODE": "intel",
               "MODEL_SUPPORTS": "stt,stt_stream", "ENGINE_ARGS": ""}
        with mock.patch.dict(os.environ, env, clear=False):
            m = importlib.reload(importlib.import_module("wrapper.caps.stt_stream"))
            self.addCleanup(lambda: _stt())
            m._resting_bytes = 0
            m._headroom_bytes = lambda: int(200.0 * 6.0 * (2 ** 20) / 0.5)
            m._scale_seen = 0
            with mock.patch.object(m.cgroup, "read",
                                   lambda: {"current": None, "max": None, "available": None}):
                self.assertAlmostEqual(m._budget_now(), 50.0, places=3)
                m._gpu_mode = lambda: "intel-gpu"
                self.assertAlmostEqual(m._budget_now(), 200.0, places=3)
                m._gpu_mode = lambda: "nvidia"
                self.assertAlmostEqual(m._budget_now(), 200.0, places=3)

    def test_a_plan_that_still_fits_is_left_alone(self):
        m = self._module(200.0)
        todo = [[(i, None, 4.0) for i in range(4)]]
        self.assertIsNone(m._replan_over_budget(todo))


class SmallerPlanLooksAtTheLargestGroupTest(unittest.TestCase):
    """🔴 `pack` sorts longest-span-first, so the FIRST group is headed by the longest span and
    therefore usually holds the FEWEST spans. Reading it said "smaller" while a later group of
    short spans grew -- and the count is the axis the cost model does not price, which is the
    axis a refusal caused by span count came from."""

    def _module(self, budget_seconds):
        m = _stt()
        m._resting_bytes = 0
        m._headroom_bytes = lambda: int(budget_seconds * 6.0 * (2 ** 20) / 0.5)
        patch = mock.patch.object(m.cgroup, "read",
                                  lambda: {"current": None, "max": None, "available": None})
        patch.start()
        self.addCleanup(patch.stop)
        return m

    def test_a_re_plan_whose_biggest_group_grew_is_not_smaller(self):
        # Refused 3 spans; the re-plan's first group is 1 span, and a later one is 20.
        m = self._module(100.0)
        refused = [(i, None, 100.0) for i in range(3)]
        todo = [[(10 + i, None, 5.0) for i in range(10)],
                [(20 + i, None, 5.0) for i in range(10)]]
        self.assertIsNone(m._smaller_plan(todo, refused),
                          "a plan with a 20-span group passed as smaller than a 3-span one")

    def test_a_re_plan_that_is_smaller_everywhere_still_goes_through(self):
        m = self._module(240.0)
        refused = [(i, None, 120.0) for i in range(4)]
        out = m._smaller_plan([], refused)
        self.assertIsNotNone(out)
        groups, _how = out
        self.assertLess(max(len(g) for g in groups), len(refused))


class HostBoundsACountAloneTest(unittest.TestCase):
    """🔴 The one place the "never stand in for a grant" rule does NOT apply, and why.

    `auto` SIZES from the budget, so a host account standing in for an unreadable grant sizes a
    batch on a card this engine has no authority over -- that is the regression this line has
    already had once. A count has fixed the size before the accounts are consulted, so the host
    is only trimming, and trimming is what kept the previous runtime's container alive when
    somebody set a count too high.
    """

    def _module(self, args, host_limit):
        m = _stt(args)
        m._resting_bytes = 0
        m._headroom_bytes = lambda: None          # no GPU authority at all
        patch = mock.patch.object(m.cgroup, "read",
                                  lambda: {"current": 0, "max": host_limit,
                                           "available": None, "reclaimable": 0})
        patch.start()
        self.addCleanup(patch.stop)
        return m

    def test_a_count_without_a_grant_builds_nothing_rather_than_being_trimmed(self):
        # 🔴 This changed with the flag surface and it is the change worth knowing about. A
        # count used to SELECT a mode that grouped by arrival, and the container's account
        # was the only thing trimming it. A count is a ceiling now, and a ceiling on a
        # machine with no readable grant leaves one span a call -- slower, and it says why,
        # rather than a batch sized by an account that was never a GPU authority.
        m = self._module("--batch-max-spans 200", 8 * (2 ** 20))
        groups, how = m._plan_groups([(i, [], 30.0) for i in range(200)])
        self.assertEqual(len(groups), 200)
        self.assertEqual(how, "one span a call")

    def test_auto_still_refuses_to_size_from_the_container_alone(self):
        m = self._module("", 8 * (2 ** 30))
        groups, how = m._plan_groups([(i, [], 30.0) for i in range(20)])
        self.assertEqual([len(g) for g in groups], [1] * 20)
        self.assertEqual(how, "one span a call")


class DegradationSaysWhichOneTest(unittest.TestCase):
    """🔴 Four ways `auto` ends up at one span a call, and they need four different answers from
    an operator: two are permanent for the life of the process and no flag touches them, one is
    a chart edit, one clears by itself. A single "could not size" line would send somebody to
    edit a quota on a machine that has no CUDA device at all."""

    NO_CGROUP = {"current": None, "max": None, "available": None}
    FULL_CGROUP = {"current": 0, "max": 0, "available": None, "reclaimable": 0}

    def _said(self, setup, cgroup):
        m = _stt()
        setup(m)
        said = []
        with mock.patch.object(m, "_p", said.append), \
             mock.patch.object(m.cgroup, "read", lambda: dict(cgroup)):
            budget = m._effective_budget()
        return budget, " ".join(said)

    def test_no_counter_names_the_machine_not_the_chart(self):
        def setup(m):
            m._resting_bytes = None
            m._held_bytes = lambda: None
            m._headroom_bytes = lambda: None
        budget, said = self._said(setup, self.NO_CGROUP)
        self.assertIsNone(budget)
        self.assertIn("no per-process memory counter", said)
        self.assertNotIn("REQUIRED_GPU_MEMORY", said)

    def test_no_grant_names_the_chart_not_the_machine(self):
        def setup(m):
            m._resting_bytes = 0
            m._held_bytes = lambda: 0
            m._headroom_bytes = lambda: None
        budget, said = self._said(setup, self.NO_CGROUP)
        self.assertIsNone(budget)
        self.assertIn("REQUIRED_GPU_MEMORY", said)
        self.assertNotIn("no per-process memory counter", said)

    def test_a_spent_grant_is_zero_and_says_it_lifts(self):
        def setup(m):
            m._resting_bytes = 0
            m._held_bytes = lambda: 0
            m._headroom_bytes = lambda: 0
        budget, said = self._said(setup, self.NO_CGROUP)
        self.assertEqual(budget, 0.0)
        self.assertIn("already spent", said)
        self.assertIn("lifts", said)

    def test_a_full_container_is_named_separately_from_a_full_card(self):
        # 🔴 The two zeros again, one level up: both are "0.0, one span a call", and they are
        # different machines to go and look at.
        def setup(m):
            m._resting_bytes = 0
            m._headroom_bytes = lambda: 8 * (2 ** 30)
        budget, said = self._said(setup, self.FULL_CGROUP)
        self.assertEqual(budget, 0.0)
        self.assertIn("container is at its memory limit", said)
        self.assertNotIn("already spent", said)


class OffMeansNoBudgetAtAllTest(unittest.TestCase):
    """🔴 Pinned rather than argued from unreachability. Today the handler's `if GROUPING:` gate
    means nothing downstream reads a budget when batching is off, so a budget solved there
    changes nothing -- but that is a fact about one call site, and the next reader of
    `_budget_now` has no way to know it. The contract is that off has no budget."""

    def test_nothing_is_solved_when_batching_is_off(self):
        m = _stt("--batch-max-spans 1")
        m._resting_bytes = 0
        m._headroom_bytes = lambda: 8 * (2 ** 30)
        with mock.patch.object(m.cgroup, "read",
                               lambda: {"current": 0, "max": 8 * (2 ** 30),
                                        "available": None, "reclaimable": 0}):
            self.assertIsNone(m._budget_now())
            self.assertIsNone(m._effective_budget())


class OneReadingPerPlanTest(unittest.TestCase):
    """🔴 The container's account is read once per plan, and the line names the reading the plan
    was made against. Reading it a second time puts two moments in one sentence -- the same shape
    an existing test catches one level up in the health line, and it says "bounded by the
    container" about an account that had nothing to do with the groups that were built."""

    def test_the_line_names_the_account_the_plan_was_solved_from(self):
        m = _stt()
        m._resting_bytes = 0
        m._headroom_bytes = lambda: int(600.0 * 6.0 * (2 ** 20) / 0.5)
        # Generous first, then tiny: a second read lands on the tiny one and the line flips.
        readings = [{"current": 0, "max": 64 * (2 ** 30), "available": None, "reclaimable": 0},
                    {"current": 0, "max": 1 * (2 ** 20), "available": None, "reclaimable": 0}]
        with mock.patch.object(m.cgroup, "read",
                               lambda: readings.pop(0) if readings else readings):
            _groups, how = m._plan_groups([(i, None, 30.0) for i in range(8)])
        self.assertNotIn("bounded by the container", how,
                         "the line named an account the plan was not solved from")
