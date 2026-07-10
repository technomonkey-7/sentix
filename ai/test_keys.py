import unittest

from ai.sentiment_analyzer import _should_rotate


class TestKeyRotationClassifier(unittest.TestCase):
    def test_quota_errors_rotate(self):
        self.assertTrue(_should_rotate(Exception("429 RESOURCE_EXHAUSTED: Quota exceeded")))
        self.assertTrue(_should_rotate(Exception("ResourceExhausted: rate limit")))
        self.assertTrue(_should_rotate(Exception("You exceeded your current quota")))

    def test_bad_key_errors_rotate(self):
        self.assertTrue(_should_rotate(Exception("400 API key not valid. Please pass a valid API key.")))
        self.assertTrue(_should_rotate(Exception("API_KEY_INVALID")))
        self.assertTrue(_should_rotate(Exception("403 PERMISSION_DENIED")))
        self.assertTrue(_should_rotate(Exception("API key expired")))

    def test_other_errors_do_not_rotate(self):
        self.assertFalse(_should_rotate(Exception("500 Internal error")))
        self.assertFalse(_should_rotate(Exception("Invalid JSON in response")))
        self.assertFalse(_should_rotate(Exception("Connection timed out")))
        self.assertFalse(_should_rotate(Exception("model not found: gemini-x")))


if __name__ == "__main__":
    unittest.main()
