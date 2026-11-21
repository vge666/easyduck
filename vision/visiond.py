#!/usr/bin/env python3
"""Optional YOLO sidecar. Sends greet, pickup, and stop through robotd IPC."""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from behave import Config, Memory, decide
from detect import open_detector
from ipc import IpcError, RobotClient


def log(message: str):
    print(f'visiond: {message}', file=sys.stderr, flush=True)


def load_capture():
    candidates = [Path(__file__).resolve().parents[1] / 'web', Path('/opt/robot/xgoduck-web')]
    for path in candidates:
        if (path / 'usb_camera.py').is_file():
            sys.path.insert(0, str(path))
            from usb_camera import capture_frames
            return capture_frames
    raise RuntimeError('usb_camera.py was not found next to the repository or in /opt/robot/xgoduck-web')


def decode_jpeg(payload: bytes):
    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is not None:
        import numpy as np
        image = Image.open(io.BytesIO(payload)).convert('RGB')
        return np.asarray(image)
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('JPEG decode needs pillow or opencv-python') from exc
    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError('camera frame was not a JPEG')
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='YOLO vision sidecar for EasyDuck')
    parser.add_argument('--socket', default='/run/robotd.sock')
    parser.add_argument('--model', default='/var/lib/robot/vision/yolo26n.onnx')
    parser.add_argument('--backend', choices=('auto', 'ultralytics', 'onnx'), default='auto')
    parser.add_argument('--imgsz', type=int, default=320)
    parser.add_argument('--conf', type=float, default=0.35)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--greet-cooldown', type=float, default=8.0)
    parser.add_argument('--pick-cooldown', type=float, default=6.0)
    return parser.parse_args(argv)


def run(args, frames, detector, client):
    memory = Memory()
    config = Config(det_conf=args.conf, greet_cooldown_s=args.greet_cooldown,
                    pick_cooldown_s=args.pick_cooldown)
    last_error = ''
    last_sent = {}
    for jpeg in frames:
        image = decode_jpeg(jpeg) if isinstance(jpeg, (bytes, bytearray)) else jpeg
        now = time.monotonic()
        detections = detector.detect(image)
        for intent in decide(detections, now, memory, config):
            # look and stop are levels. Refresh inside the daemon's one-second hold
            # without a request on every camera frame.
            if intent.method in ('robot.look', 'robot.stop') and now - last_sent.get(intent.method, 0.0) < 0.2:
                continue
            try:
                client.call(intent.method, intent.params)
                last_sent[intent.method] = now
                if intent.method != 'robot.look':
                    log(f'{intent.method} {intent.params}')
                last_error = ''
            except IpcError as exc:
                text = str(exc)
                if text != last_error:
                    log(text)
                    last_error = text


def main(argv=None):
    args = parse_args(argv)
    model = Path(args.model)
    if not model.is_file():
        log(f'model not found: {model}')
        log('export yolo26n on a machine with ultralytics, then copy the .onnx file; see docs/en/vision.md')
        return 1
    if args.imgsz < 32:
        log('--imgsz must be at least 32')
        return 2
    try:
        detector = open_detector(model, args.backend, args.imgsz, args.conf, args.device, args.threads)
    except Exception as exc:
        log(f'detector failed to load: {exc}')
        return 1
    client = RobotClient(args.socket, dry_run=args.dry_run)
    try:
        with load_capture()() as frames:
            run(args, frames, detector, client)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        log(str(exc))
        return 1
    finally:
        client.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
