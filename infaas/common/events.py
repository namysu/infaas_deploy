"""JSON-lines logs for experiments: one line per request, one per scaling/state event.

Written off the request path by a background thread. Paths come from
REQUEST_LOG / EVENT_LOG; empty disables the file (the log line is still emitted
at DEBUG for requests and INFO for events).
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from typing import Optional

from infaas.common import config

log = logging.getLogger("events")


class _Writer:
    def __init__(self, path: str) -> None:
        self.path = path
        self.q: "queue.Queue[str]" = queue.Queue(maxsize=100000)
        if path:
            threading.Thread(target=self._run, daemon=True, name=f"log:{path}").start()

    def put(self, rec: dict) -> None:
        if not self.path:
            return
        try:
            self.q.put_nowait(json.dumps(rec, separators=(",", ":")))
        except queue.Full:
            pass

    def _run(self) -> None:
        with open(self.path, "a", buffering=1) as f:
            while True:
                line = self.q.get()
                f.write(line + "\n")


_requests: Optional[_Writer] = None
_events: Optional[_Writer] = None
_lock = threading.Lock()


def _get(kind: str) -> _Writer:
    global _requests, _events
    with _lock:
        if kind == "req":
            if _requests is None:
                _requests = _Writer(config.REQUEST_LOG)
            return _requests
        if _events is None:
            _events = _Writer(config.EVENT_LOG)
        return _events


def request(**fields) -> None:
    fields.setdefault("ts", time.time())
    _get("req").put(fields)


def event(kind: str, **fields) -> None:
    fields["kind"] = kind
    fields.setdefault("ts", time.time())
    log.info("event %s", json.dumps(fields, separators=(",", ":")))
    _get("evt").put(fields)
