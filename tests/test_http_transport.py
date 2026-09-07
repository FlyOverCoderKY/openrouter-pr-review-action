"""Exercise real sockets, including responses that never hit an idle timeout."""

import io
import json
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from or_pr_review import http_transport
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


@pytest.mark.parametrize(
    "reason",
    [
        TimeoutError("timed out"),
        ConnectionRefusedError(111, "Connection refused"),
        socket.gaierror(-2, "Name or service not known"),
    ],
)
def test_worker_preserves_wrapped_error_classification_and_reason(monkeypatch, reason):
    payload = {
        "url": "https://example.test/",
        "method": "POST",
        "headers": {"Authorization": "Bearer test-secret"},
        "body": "cHJpdmF0ZSBwcm9tcHQ=",
        "timeout": 5,
    }
    output = io.StringIO()
    monkeypatch.setattr(http_transport.sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(http_transport.sys, "stdout", output)

    def fail(*args, **kwargs):
        raise urllib.error.URLError(reason)

    monkeypatch.setattr(http_transport.urllib.request, "urlopen", fail)
    http_transport.main()
    serialized = output.getvalue()
    reply = json.loads(serialized)
    assert "test-secret" not in serialized
    assert "private prompt" not in serialized
    assert payload["body"] not in serialized

    # Exercise reconstruction with exactly the response emitted by the worker.
    def completed(*args, **kwargs):
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(args[0], 0, serialized, "")

    monkeypatch.setattr(http_transport.subprocess, "run", completed)
    request = urllib.request.Request(payload["url"])
    if isinstance(reason, TimeoutError):
        assert reply == {"kind": "timeout"}
        with pytest.raises(TimeoutError, match="HTTP request timed out"):
            bounded_urlopen(request, timeout=5)
    else:
        assert reply == {
            "kind": "connection",
            "category": type(reason).__name__,
            "reason": str(reason),
        }
        with pytest.raises(urllib.error.URLError) as caught:
            bounded_urlopen(request, timeout=5)
        assert caught.value.reason == f"{type(reason).__name__}: {reason}"


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
