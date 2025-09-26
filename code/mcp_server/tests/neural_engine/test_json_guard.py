import json
import unittest

from mcp_server.neural_engine.json_guard import ensure_json


class TestJsonGuard(unittest.TestCase):
    def test_parse_then_retry_then_default(self):
        # Case 1: first call returns good JSON
        res = ensure_json(lambda: '{"a":1}')
        self.assertEqual(res.get('a'), 1)

        # Case 2: first bad, second good
        calls = {'n': 0}
        def first_then_good():
            calls['n'] += 1
            return 'not json' if calls['n'] == 1 else '{"ok":true}'
        r2 = ensure_json(first_then_good)
        self.assertTrue(r2.get('ok'))

        # Case 3: both bad -> default
        d = ensure_json(lambda: 'bad', lambda: 'still bad', default_factory=lambda: {"fallback": True})
        self.assertTrue(d.get('fallback'))

