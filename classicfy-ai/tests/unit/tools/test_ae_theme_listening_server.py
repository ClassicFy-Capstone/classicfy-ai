"""Canonical playback origin and non-destructive browser response transfer."""

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from urllib.parse import urlencode

AI = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("ae_theme_server", AI / "tools/ae_theme_listening/serve.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class ListeningServerTest(unittest.TestCase):
    def body(self, state=None, path="/?qa=1#section"):
        return urlencode({"state": json.dumps(state or {server.STORAGE_PREFIX + "pack": "saved response"}), "next": path}).encode()

    def handler(self, body, **headers):
        handler = object.__new__(server.ListeningHandler)
        handler.path = "/__origin_transfer"
        handler.server = SimpleNamespace(server_port=8766)
        handler.headers = {"Host": "localhost:8766", "Origin": "http://127.0.0.1:8766",
                           "Content-Type": "application/x-www-form-urlencoded", "Content-Length": str(len(body)), **headers}
        handler.rfile, handler.wfile = io.BytesIO(body), io.BytesIO()
        handler.response_status = None
        handler.response_headers = {}
        handler.send_response = lambda value: setattr(handler, "response_status", value)
        handler.send_error = handler.send_response
        handler.send_header = lambda key, value: handler.response_headers.update({key: value})
        handler.end_headers = lambda: None
        return handler

    def test_owned_storage_round_trips_without_parsing_or_rewriting_listener_answers(self):
        state = {server.STORAGE_PREFIX + "pack": json.dumps({"memo": "기존 메모", "xy_order": ["c", "b"]})}
        self.assertEqual(server.parse_transfer(self.body(state)), (state, "/?qa=1#section"))

    def test_transfer_rejects_unrelated_browser_storage_and_ambiguous_fields(self):
        for body in (self.body({"unrelated-session": "private"}), self.body({server.STORAGE_PREFIX + "pack": {"object": True}}),
                     self.body() + b"&state=%7B%7D", b"state=%5B%5D&next=%2F"):
            with self.assertRaises(ValueError):
                server.parse_transfer(body)

    def test_return_path_stays_on_the_local_application(self):
        for path in ("https://example.com/", "//example.com/", "/\\example.com", "/\nheader", "relative"):
            with self.assertRaises(ValueError):
                server.parse_transfer(self.body(path=path))

    def test_post_accepts_only_the_old_local_origin_and_canonical_host(self):
        good = self.handler(self.body())
        good.do_POST()
        self.assertEqual(good.response_status, 200)
        self.assertEqual(good.response_headers["Cache-Control"], "no-store")
        for headers, status in [({"Origin": "https://example.com"}, 403), ({"Host": "127.0.0.1:8766"}, 403),
                                ({"Content-Type": "application/json"}, 415), ({"Content-Length": str(server.MAX_TRANSFER_BYTES + 1)}, 413)]:
            handler = self.handler(self.body(), **headers)
            handler.do_POST()
            self.assertEqual(handler.response_status, status)

    def test_script_data_cannot_close_script_or_change_template_tokens(self):
        memo = "</script><script>alert(1)</script> __NEXT__ $next"
        page = server.transfer_page({server.STORAGE_PREFIX + "pack": memo}, "/?qa=1")
        self.assertNotIn(memo, page)
        self.assertIn("\\u003c/script\\u003e", page)
        self.assertIn("__NEXT__ $next", page)
        self.assertEqual(page.count("</script>"), 1)

    def test_ip_entry_transfers_owned_keys_without_deleting_the_originals(self):
        handler = self.handler(b"", Host="127.0.0.1:8766")
        handler.path = "/?qa=1"
        handler.do_GET()
        page = handler.wfile.getvalue().decode()
        self.assertEqual(handler.response_status, 200)
        self.assertIn("http://localhost:8766", page)
        self.assertIn(server.STORAGE_PREFIX, page)
        self.assertNotIn("removeItem", page)
        self.assertNotIn("localStorage.clear", page)


if __name__ == "__main__":
    unittest.main()
