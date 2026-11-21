"""Map YOLO detections to existing robotd intents.

Coordinates follow the camera image (x right, y down) and the daemon's trunk
frame (x forward, y left, z up). A person on the right side of the image is
to the robot's right, so the gaze ``y`` is negative.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from detect import Detection


PERSON = 0
# COCO: sports ball, bottle, cup, banana, apple, orange, cell phone, book.
SMALL = frozenset({32, 39, 41, 46, 47, 49, 67, 73})
LOOK_X = 1.0
LOOK_Y_GAIN = 0.55
LOOK_Z_GAIN = 0.30


@dataclass(frozen=True)
class Intent:
    method: str
    params: dict


@dataclass
class Config:
    person_conf: float = 0.45
    det_conf: float = 0.35
    forward_x: tuple[float, float] = (0.30, 0.70)
    forward_y: tuple[float, float] = (0.42, 0.98)
    obstacle_area: float = 0.08
    person_obstacle_area: float = 0.20
    small_min_area: float = 0.004
    small_max_area: float = 0.12
    small_min_y: float = 0.35
    greet_cooldown_s: float = 8.0
    pick_cooldown_s: float = 6.0
    mouth_open: float = 0.8


@dataclass
class Memory:
    greeted: dict = field(default_factory=dict)
    picked: dict = field(default_factory=dict)


def track_key(det: Detection):
    if det.track is not None:
        return ('id', det.cls, det.track)
    return ('cell', det.cls, int(det.cx * 4), int(det.cy * 4))


def gaze(det: Detection) -> tuple[float, float, float]:
    nx = (det.cx - 0.5) * 2
    ny = (det.cy - 0.5) * 2
    y = max(-0.8, min(0.8, -nx * LOOK_Y_GAIN))
    z = max(-0.45, min(0.45, -ny * LOOK_Z_GAIN))
    return (LOOK_X, y, z)


def _due(store: dict, key, now: float, cooldown: float) -> bool:
    last = store.get(key)
    return last is None or now - last >= cooldown


def _forward(det: Detection, config: Config) -> bool:
    return (config.forward_x[0] <= det.cx <= config.forward_x[1]
            and config.forward_y[0] <= det.cy <= config.forward_y[1])


def is_obstacle(det: Detection, config: Config) -> bool:
    if det.conf < config.det_conf or not _forward(det, config):
        return False
    if det.cls == PERSON:
        return det.area >= config.person_obstacle_area
    if det.cls in SMALL and det.area < config.small_max_area:
        return False
    return det.area >= config.obstacle_area


def decide(detections: list[Detection], now: float, memory: Memory, config: Config | None = None) -> list[Intent]:
    config = config or Config()
    people = [det for det in detections if det.cls == PERSON and det.conf >= config.person_conf]
    person = max(people, key=lambda det: det.area) if people else None
    obstacle = any(is_obstacle(det, config) for det in detections)
    small = [det for det in detections
             if det.cls in SMALL and det.conf >= config.det_conf
             and config.small_min_area <= det.area < config.small_max_area
             and det.cy >= config.small_min_y]
    target = max(small, key=lambda det: det.cy) if small else None
    intents = []
    if obstacle:
        intents.append(Intent('robot.stop', {}))
    if person is not None:
        x, y, z = gaze(person)
        intents.append(Intent('robot.look', {'x': x, 'y': y, 'z': z, 'neck_pitch': 0.0}))
        key = track_key(person)
        if _due(memory.greeted, key, now, config.greet_cooldown_s):
            memory.greeted[key] = now
            intents.append(Intent('robot.sound', {'tag': 'greet'}))
            intents.append(Intent('robot.mouth', {'open': config.mouth_open}))
    if target is not None and not obstacle:
        key = track_key(target)
        if _due(memory.picked, key, now, config.pick_cooldown_s):
            memory.picked[key] = now
            intents.append(Intent('robot.do', {'skill': 'ground_pick'}))
    return intents
