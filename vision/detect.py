"""YOLO detect/track for the vision sidecar.

The Ultralytics package (https://github.com/vge666/ultralytics) is used when a
``.pt`` model is requested. ONNX Runtime decodes an exported YOLO26 end-to-end
head, or a classic ``(4 + classes, anchors)`` head, when PyTorch is not installed.
Neither path fabricates boxes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import sys

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One box in normalized image coordinates, origin at the top left."""

    cx: float
    cy: float
    w: float
    h: float
    conf: float
    cls: int
    track: int | None = None

    @property
    def area(self) -> float:
        return self.w * self.h

    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.cx - self.w / 2, self.cy - self.h / 2,
                self.cx + self.w / 2, self.cy + self.h / 2)

    def with_track(self, track: int) -> 'Detection':
        return replace(self, track=track)


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    union += max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union -= inter
    return inter / union if union > 0 else 0.0


class Tracker:
    """Greedy IoU association used when Ultralytics tracking ids are absent."""

    def __init__(self, min_iou: float = 0.3, max_missed: int = 8):
        self.min_iou = min_iou
        self.max_missed = max_missed
        self.tracks: list[dict] = []
        self.next_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                if det.cls != track['cls']:
                    continue
                score = iou(track['box'], det.xyxy())
                if score >= self.min_iou:
                    pairs.append((score, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d, matched = set(), set(), {}
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matched[di] = self.tracks[ti]['id']
            self.tracks[ti]['box'] = detections[di].xyxy()
            self.tracks[ti]['missed'] = 0
        for ti, track in enumerate(self.tracks):
            if ti not in used_t:
                track['missed'] += 1
        self.tracks = [track for track in self.tracks if track['missed'] <= self.max_missed]
        result = []
        for di, det in enumerate(detections):
            if di in matched:
                result.append(det.with_track(matched[di]))
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks.append({'id': track_id, 'cls': det.cls, 'box': det.xyxy(), 'missed': 0})
            result.append(det.with_track(track_id))
        return result


def resize_bilinear(image: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    if src_h == height and src_w == width:
        return image
    y = (np.arange(height) + 0.5) * src_h / height - 0.5
    x = (np.arange(width) + 0.5) * src_w / width - 0.5
    y0 = np.clip(np.floor(y).astype(np.int32), 0, src_h - 1)
    x0 = np.clip(np.floor(x).astype(np.int32), 0, src_w - 1)
    y1 = np.clip(y0 + 1, 0, src_h - 1)
    x1 = np.clip(x0 + 1, 0, src_w - 1)
    wy = (y - y0).astype(np.float32)[:, None, None]
    wx = (x - x0).astype(np.float32)[None, :, None]
    src = image.astype(np.float32)
    top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
    bottom = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
    return np.clip(top * (1 - wy) + bottom * wy + 0.5, 0, 255).astype(np.uint8)


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Fit ``image`` into a ``size`` square with gray 114 padding, matching Ultralytics."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('YOLO input must be an HxWx3 image')
    src_h, src_w = image.shape[:2]
    if src_h < 1 or src_w < 1 or size < 1:
        raise ValueError('empty YOLO input')
    scale = min(size / src_h, size / src_w)
    height = min(size, max(1, int(round(src_h * scale))))
    width = min(size, max(1, int(round(src_w * scale))))
    resized = resize_bilinear(image, height, width)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top = (size - height) // 2
    left = (size - width) // 2
    canvas[top:top + height, left:left + width] = resized
    return canvas, scale, left, top


def blob_from_canvas(canvas: np.ndarray, uint8: bool) -> np.ndarray:
    if uint8:
        return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


def _boxes_from_xyxy(rows: np.ndarray, scale: float, left: int, top: int,
                     width: int, height: int, conf: float) -> list[Detection]:
    found = []
    if scale <= 0 or width < 1 or height < 1:
        return found
    for x1, y1, x2, y2, score, cls in rows:
        score = float(score)
        if score < conf or not np.isfinite(score):
            continue
        class_id = int(round(float(cls)))
        if class_id < 0:
            continue
        px1 = (float(x1) - left) / scale
        py1 = (float(y1) - top) / scale
        px2 = (float(x2) - left) / scale
        py2 = (float(y2) - top) / scale
        px1, px2 = sorted((px1, px2))
        py1, py2 = sorted((py1, py2))
        px1, py1 = max(0.0, px1), max(0.0, py1)
        px2, py2 = min(float(width), px2), min(float(height), py2)
        if px2 - px1 < 1 or py2 - py1 < 1:
            continue
        found.append(Detection(
            cx=(px1 + px2) / 2 / width, cy=(py1 + py2) / 2 / height,
            w=(px2 - px1) / width, h=(py2 - py1) / height,
            conf=score, cls=class_id))
    return found


def nms(detections: list[Detection], iou_thres: float) -> list[Detection]:
    order = sorted(range(len(detections)), key=lambda i: detections[i].conf, reverse=True)
    kept = []
    while order:
        current = order.pop(0)
        kept.append(detections[current])
        order = [i for i in order
                 if detections[i].cls != detections[current].cls
                 or iou(detections[current].xyxy(), detections[i].xyxy()) <= iou_thres]
    return kept


def _classic_rows(values: np.ndarray, conf: float) -> list[tuple]:
    """``values`` is anchors × (xywh + class scores) in network-input pixels."""
    if values.shape[1] < 5:
        return []
    boxes = values[:, :4]
    scores = values[:, 4:]
    class_ids = scores.argmax(axis=1)
    best = scores[np.arange(scores.shape[0]), class_ids]
    rows = []
    for (cx, cy, w, h), score, cls in zip(boxes, best, class_ids):
        if float(score) < conf:
            continue
        rows.append((float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2),
                     float(score), int(cls)))
    return rows


def decode_array(array: np.ndarray, scale: float, left: int, top: int, width: int, height: int,
                 conf: float, iou_thres: float) -> list[Detection] | None:
    """Return detections, or ``None`` when the tensor is not a YOLO detect head."""
    values = np.asarray(array)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2 or min(values.shape) < 1:
        return None
    # YOLO26 end-to-end export is (detections, 6): xyxy, score, class.
    if values.shape[-1] == 6:
        return _boxes_from_xyxy(values, scale, left, top, width, height, conf)
    # Classic export is (4 + classes, anchors). Anchors are the longer axis.
    if values.shape[0] < values.shape[1] and values.shape[0] >= 5:
        values = values.T
    if values.shape[1] < 5:
        return None
    rows = _classic_rows(values, conf)
    decoded = _boxes_from_xyxy(np.asarray(rows, dtype=np.float32).reshape(-1, 6)
                               if rows else np.zeros((0, 6), np.float32),
                               scale, left, top, width, height, conf)
    return nms(decoded, iou_thres)


def decode_outputs(outputs, scale: float, left: int, top: int, width: int, height: int,
                   conf: float, iou_thres: float) -> list[Detection]:
    recognized = []
    for output in outputs:
        decoded = decode_array(output, scale, left, top, width, height, conf, iou_thres)
        if decoded is not None:
            recognized.append(decoded)
    if not recognized:
        raise ValueError('YOLO output is not an end-to-end (N, 6) head or a classic detect head')
    for decoded in recognized:
        if decoded:
            return decoded
    return recognized[0]


class OnnxDetector:
    """ONNX Runtime detect plus the local IoU tracker."""

    def __init__(self, path, imgsz: int, conf: float, iou_thres: float = 0.45, threads: int = 2):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        self.session = ort.InferenceSession(str(path), sess_options=options,
                                            providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0]
        shape = self.input.shape
        self.size = imgsz
        if len(shape) >= 4 and isinstance(shape[2], int) and shape[2] > 0:
            self.size = int(shape[2])
        if len(shape) >= 4 and isinstance(shape[3], int) and shape[3] > 0 and shape[3] != self.size:
            raise RuntimeError('non-square YOLO ONNX input is not supported')
        self.conf = conf
        self.iou_thres = iou_thres
        self.uint8 = 'uint8' in self.input.type
        self.tracker = Tracker()

    def detect(self, image: np.ndarray) -> list[Detection]:
        image = np.asarray(image)
        canvas, scale, left, top = letterbox(image, self.size)
        blob = blob_from_canvas(canvas, self.uint8)
        outputs = self.session.run(None, {self.input.name: blob})
        height, width = image.shape[:2]
        found = decode_outputs(outputs, scale, left, top, width, height, self.conf, self.iou_thres)
        return self.tracker.update(found)


class UltralyticsDetector:
    """``ultralytics.YOLO.track`` (ByteTrack), with ``predict`` if tracking cannot start."""

    def __init__(self, path, imgsz: int, conf: float, device: str = 'cpu'):
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self.track = True
        self.tracker = Tracker()

    def detect(self, image: np.ndarray) -> list[Detection]:
        image = np.asarray(image)
        height, width = image.shape[:2]
        kwargs = dict(verbose=False, conf=self.conf, imgsz=self.imgsz, device=self.device)
        if self.track:
            try:
                results = self.model.track(image, persist=True, tracker='bytetrack.yaml', **kwargs)
            except Exception as exc:
                self.track = False
                print(f'visiond: YOLO.track failed ({exc}); using predict', file=sys.stderr, flush=True)
                results = self.model.predict(image, **kwargs)
        else:
            results = self.model.predict(image, **kwargs)
        if not isinstance(results, list):
            results = list(results)
        found = _from_ultralytics(results[0], width, height)
        if found and all(det.track is not None for det in found):
            return found
        return self.tracker.update([replace(det, track=None) for det in found])


def _from_ultralytics(result, width: int, height: int) -> list[Detection]:
    boxes = getattr(result, 'boxes', None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy() if hasattr(boxes.xyxy, 'detach') else np.asarray(boxes.xyxy)
    conf = boxes.conf.detach().cpu().numpy() if hasattr(boxes.conf, 'detach') else np.asarray(boxes.conf)
    cls = boxes.cls.detach().cpu().numpy() if hasattr(boxes.cls, 'detach') else np.asarray(boxes.cls)
    ids = None
    if getattr(boxes, 'id', None) is not None:
        ids = boxes.id.detach().cpu().numpy() if hasattr(boxes.id, 'detach') else np.asarray(boxes.id)
    found = []
    for index, (box, score, class_id) in enumerate(zip(xyxy, conf, cls)):
        x1, y1, x2, y2 = [float(v) for v in box]
        if width < 1 or height < 1 or x2 <= x1 or y2 <= y1:
            continue
        track = None if ids is None else int(ids[index])
        found.append(Detection(
            cx=(x1 + x2) / 2 / width, cy=(y1 + y2) / 2 / height,
            w=(x2 - x1) / width, h=(y2 - y1) / height,
            conf=float(score), cls=int(class_id), track=track))
    return found


def open_detector(path, backend: str, imgsz: int, conf: float, device: str, threads: int):
    suffix = str(path).lower()
    if backend == 'auto':
        backend = 'ultralytics' if suffix.endswith('.pt') else 'onnx'
    if backend == 'ultralytics':
        return UltralyticsDetector(path, imgsz, conf, device)
    if backend == 'onnx':
        return OnnxDetector(path, imgsz, conf, threads=threads)
    raise ValueError('backend must be auto, ultralytics, or onnx')
