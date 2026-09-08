"""HTTP attempts isolated so DNS, headers and trickling bodies share a hard deadline."""

from __future__ import annotations

import base64
import http.client
import io
import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path


class HttpElapsedTimeout(TimeoutError):
    """The worker reached its absolute elapsed-time limit."""


class HttpIdleTimeout(TimeoutError):
    """A socket operation timed out while waiting for activity."""


class HttpConnectTimeout(TimeoutError):
    """DNS, connection or response headers exceeded their watchdog limit."""


_CONNECT_TIMEOUT_EXIT = 124


def bounded_urlopen(
    request: urllib.request.Request, *, timeout: float, total_timeout: float | None = None
) -> io.BytesIO:
    """Buffer a response with socket inactivity and hard elapsed-time limits.

    Callers without a separate total retain the original elapsed-time limit.
    Credentials travel only over stdin.
    """
    elapsed_limit = timeout if total_timeout is None else total_timeout
    payload = {
        "url": request.full_url,
        "method": request.get_method(),
        "headers": dict(request.header_items()),
        "body": base64.b64encode(request.data or b"").decode("ascii"),
        "timeout": min(timeout, elapsed_limit),
    }
    try:
        result = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve())],
            input=json.dumps(payload),
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=elapsed_limit,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
    except subprocess.TimeoutExpired:
        # subprocess.run kills and reaps the worker, including its open socket.
        # Never propagate TimeoutExpired: its payload can contain credentials.
        raise HttpElapsedTimeout("HTTP attempt elapsed-time deadline exhausted") from None
    if result.returncode == _CONNECT_TIMEOUT_EXIT:
        raise HttpConnectTimeout("HTTP connection/header phase timed out")
    if result.returncode:
        raise OSError("HTTP transport worker failed")
    reply = json.loads(result.stdout)
    kind = reply["kind"]
    if kind == "timeout":
        raise HttpIdleTimeout("HTTP request timed out")
    if kind == "connection":
        # The harness redacts this reason before publishing diagnostics.
        raise urllib.error.URLError(f"{reply['category']}: {reply['reason']}")
    body = io.BytesIO(base64.b64decode(reply["body"]))
    if kind == "http_error":
        headers = Message()
        for key, value in reply["headers"]:
            headers[key] = value
        raise urllib.error.HTTPError(request.full_url, reply["status"], "HTTP error", headers, body)
    return body


def main() -> None:
    payload = json.load(sys.stdin)
    request = urllib.request.Request(
        payload["url"],
        data=base64.b64decode(payload["body"]),
        headers=payload["headers"],
        method=payload["method"],
    )
    # Socket timeouts cannot interrupt getaddrinfo. Bound connection/header
    # setup separately before allowing an active body the longer stage budget.
    watchdog = threading.Timer(payload["timeout"], lambda: os._exit(_CONNECT_TIMEOUT_EXIT))
    watchdog.daemon = True
    watchdog.start()
    try:
        try:
            with urllib.request.urlopen(request, timeout=payload["timeout"]) as response:
                watchdog.cancel()
                reply = {"kind": "ok", "body": base64.b64encode(response.read()).decode("ascii")}
        except urllib.error.HTTPError as exc:
            watchdog.cancel()
            with exc:
                reply = {
                    "kind": "http_error",
                    "status": exc.code,
                    "headers": list(exc.headers.items()),
                    "body": base64.b64encode(exc.read()).decode("ascii"),
                }
    except TimeoutError:
        reply = {"kind": "timeout"}
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        # urllib wraps connect/TLS/send failures, including socket timeouts.
        # Preserve their classification rather than reporting all as connections.
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        if isinstance(reason, TimeoutError):
            reply = {"kind": "timeout"}
        else:
            # Send only the exception reason, never the request or payload. This
            # private IPC is consumed by the harness's redacting URLError branch.
            reply = {
                "kind": "connection",
                "category": type(reason).__name__,
                "reason": str(reason),
            }
    finally:
        watchdog.cancel()
    json.dump(reply, sys.stdout)


if __name__ == "__main__":
    main()
