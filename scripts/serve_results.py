#!/usr/bin/env python3
"""Serve /work/out on port 8000, but only to requests that carry the bearer token in $LAYA_TOKEN.

Runs on the Runpod pod after training. The token is read from the environment and never printed or
written to train.log. Requests without `Authorization: Bearer <token>` get 401.
"""
from __future__ import annotations

import functools
import hmac
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

TOKEN = os.environ.pop("LAYA_TOKEN", "")
if len(TOKEN) < 32:
    sys.exit("LAYA_TOKEN missing or too short; refusing to serve results")
EXPECTED = f"Bearer {TOKEN}".encode()


class AuthHandler(SimpleHTTPRequestHandler):
    def _authorized(self) -> bool:
        return hmac.compare_digest(self.headers.get("Authorization", "").encode(), EXPECTED)

    def do_GET(self) -> None:
        if not self._authorized():
            self.send_error(401, "Unauthorized")
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if not self._authorized():
            self.send_error(401, "Unauthorized")
            return
        super().do_HEAD()

    def list_directory(self, path):  # no directory listings
        self.send_error(404, "Not found")
        return None

    def log_message(self, fmt, *args):  # keep request lines (and headers) out of logs
        pass


if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else "/work/out"
    handler = functools.partial(AuthHandler, directory=directory)
    ThreadingHTTPServer(("0.0.0.0", 8000), handler).serve_forever()
