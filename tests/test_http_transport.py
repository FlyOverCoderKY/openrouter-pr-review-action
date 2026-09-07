"""Exercise real sockets, including responses that never hit an idle timeout."""

import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from or_pr_review.http_transport import bounded_urlopen


@contextmanager
def server(handler):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


class QuietHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass


@pytest.mark.parametrize("phase", ["headers", "body", "error-body"])
def test_elapsed_deadline_kills_trickling_response(phase):
    disconnected = threading.Event()

    class Handler(QuietHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if phase != "headers":
                self.send_response(429 if phase == "error-body" else 200)
                self.end_headers()
            try:
                # Each byte arrives well inside the socket's idle timeout.
                for _ in range(100):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                disconnected.set()

    with server(Handler) as url:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="elapsed-time"):
            bounded_urlopen(urllib.request.Request(url, data=b"{}"), timeout=0.8)
        assert time.monotonic() - started < 2
        # The timed-out request is terminated, not left running in a thread.
        assert disconnected.wait(2)


@pytest.mark.parametrize("status", [200, 429])
def test_transport_preserves_body_and_retry_headers(status):
    class Handler(QuietHandler):
        def do_POST(self):
            assert self.headers["Authorization"] == "Bearer test-secret"
            assert self.rfile.read(int(self.headers["Content-Length"])) == b"{}"
            self.send_response(status)
            self.send_header("Retry-After", "7")
            self.end_headers()
            self.wfile.write(b'{"provider":"test","choices":[]}')

    with server(Handler) as url:
        request = urllib.request.Request(
            url, data=b"{}", headers={"Authorization": "Bearer test-secret"}
        )
        if status == 429:
            with pytest.raises(urllib.error.HTTPError) as caught:
                bounded_urlopen(request, timeout=5)
            assert caught.value.code == 429
            assert caught.value.headers["Retry-After"] == "7"
            assert b'"provider":"test"' in caught.value.read()
        else:
            with bounded_urlopen(request, timeout=5) as response:
                assert b'"choices":[]' in response.read()
