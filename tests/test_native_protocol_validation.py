import json
import unittest

from fastapi import HTTPException
from wrapper.caps import s3_s2st, seamless


class SampleRateTests(unittest.TestCase):
    def test_invalid_rate_is_rejected_instead_of_becoming_the_default(self):
        for cap in (s3_s2st, seamless):
            for rate in (0, -1, "invalid"):
                with self.subTest(cap=cap.__name__, rate=rate):
                    with self.assertRaises(HTTPException) as caught:
                        cap._ws_text(json.dumps({"sample_rate": rate}), "eng", "cmn",
                                     cap._new_states(), 16000)
                    self.assertEqual(caught.exception.status_code, 400)

    def test_omitted_rate_keeps_the_negotiated_rate(self):
        for cap in (s3_s2st, seamless):
            result = cap._ws_text("{}", "eng", "cmn", cap._new_states(), 24000)
            self.assertEqual(result[3], 24000)


if __name__ == "__main__":
    unittest.main()
