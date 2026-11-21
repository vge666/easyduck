"""Newline-delimited JSON-RPC client for ``/run/robotd.sock``.

Same envelope as the web gateway. Motion is not retried: a lost reply is an
error, and the next camera frame decides again.
"""
from __future__ import annotations

import json
import socket
import threading


class IpcError(Exception):
    pass


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


class RobotClient:
    def __init__(self, path: str, dry_run: bool = False, timeout: float = 2.0):
        self.path = path
        self.dry_run = dry_run
        self.timeout = timeout
        self.seq = 0
        self.lock = threading.Lock()
        self.sock = None
        self.reader = None
        self.sent: list[tuple[str, dict]] = []

    def close(self):
        if self.reader is not None:
            self.reader.close()
        if self.sock is not None:
            self.sock.close()
        self.sock = self.reader = None

    def call(self, method: str, params: dict | None = None):
        params = params or {}
        self.sent.append((method, params))
        if self.dry_run:
            return {'accepted': True, 'dry_run': True}
        with self.lock:
            try:
                if self.sock is None:
                    self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    self.sock.settimeout(self.timeout)
                    self.sock.connect(self.path)
                    self.reader = self.sock.makefile('rb')
                self.seq += 1
                payload = dumps({'jsonrpc': '2.0', 'id': self.seq, 'method': method, 'params': params})
                self.sock.sendall((payload + '\n').encode())
                line = self.reader.readline(1024 * 1024)
                if not line or not line.endswith(b'\n'):
                    raise OSError('robotd reply missing or too long')
                answer = json.loads(line)
                if answer.get('id') != self.seq:
                    raise OSError('robotd reply id mismatch')
                if 'error' in answer:
                    message = answer['error'].get('message', 'robotd rejected the request')
                    raise IpcError(message)
                result = answer.get('result')
                if isinstance(result, dict) and result.get('accepted') is False:
                    raise IpcError(result.get('reason') or 'robotd rejected the request')
                return result
            except (OSError, ValueError) as exc:
                self.close()
                raise IpcError(f'robotd communication failed: {exc}') from exc
