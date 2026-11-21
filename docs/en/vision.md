# YOLO vision

**English** | [简体中文](../zh-CN/vision.md)

Optional camera path. A Python sidecar runs YOLO and sends the existing `robot.look`, `robot.sound`, `robot.mouth`, `robot.do`, and `robot.stop` requests. It does not join the 50 Hz loop, open the servo bus, or replace `robotd`.

## What it does

| Seen in the camera | Request |
| --- | --- |
| Person | `robot.look` toward them, `robot.sound` tag `greet`, and `robot.mouth` |
| Small COCO object low in the frame | `robot.do` skill `ground_pick` |
| Large object in the forward part of the frame, or a person filling that view | `robot.stop` |

`greet` is the existing wake-up call. The mouth command is the same 0..1 opening the web controller already sends. Pickup is the existing ground-pick policy, which does not aim at the detected box. Stop zeros walking velocity and leaves the policy enabled.

Small classes are COCO ids 32, 39, 41, 46, 47, 49, 67, and 73 (sports ball, bottle, cup, banana, apple, orange, cell phone, book). A small object is not also treated as an obstacle. An obstacle suppresses pickup until it leaves the forward view. Greeting can still run while stopped.

The sidecar does not take browser or gamepad ownership. On the installed XGO profile, `robotd` applies a fresh look, mouth, or stop on top of the current XGO command for about 1 second, then returns to the stick or web command. The sidecar repeats the request while the camera still supports it.

## Requirements

- `robotd` already running, with the walking policy enabled before a pickup can start. A missing ground-pick model is refused with a reason.
- A `greet` sound needs the existing voice bank. Without it, `robot.sound` is refused and the mouth command can still be sent.
- A USB UVC camera that exposes MJPEG. Capture reuses `web/usb_camera.py`.
- Extra Python packages listed in `vision/requirements.lock`. They are not required by `robotd`, the web controller, or calibration.
- YOLO weights. This repository does not contain them.

The TaishanPi is ARM64 and already uses ONNX Runtime inside `robotd`. A full PyTorch install is not required on the board. Use an ONNX export and the Python `onnxruntime` package, which is separate from the native library `robotd` loads. Prefer ONNX Runtime 1.23 or newer.

The Ultralytics package is the detect/track path for a `.pt` file. Its source is [vge666/ultralytics](https://github.com/vge666/ultralytics). Upstream Ultralytics code and the released YOLO weights are AGPL-3.0 unless you have another Ultralytics license. They are not included here and are not relicensed as Apache-2.0.

## Weights

On a development machine that can install Ultralytics and PyTorch, from any directory where the `yolo` command is on `PATH`:

```sh
pip install ultralytics
yolo export model=yolo26n.pt format=onnx imgsz=320
```

`yolo26n` is the small COCO detect model. `imgsz=320` is the board default so the RK3566 keeps time for the control loop; 640 is the size the model was trained at and is heavier. Copy the resulting `yolo26n.onnx` to the board. Do not commit the file.

```sh
sudo install -d /var/lib/robot/vision
sudo install -m 0644 yolo26n.onnx /var/lib/robot/vision/yolo26n.onnx
```

`--backend auto` selects Ultralytics for a `.pt` file and ONNX Runtime for a `.onnx` file. `--backend ultralytics` calls `YOLO.track` (ByteTrack). If tracking cannot start, the sidecar uses `predict` and its own IoU tracker. The ONNX path always uses that IoU tracker. Both paths return real boxes or no boxes.

## Run

Install the extra packages on the board, then start the sidecar from the repository root. `--threads 2` leaves cores for `robotd`.

```sh
python3 -m pip install --user numpy pillow 'onnxruntime>=1.23'
python3 vision/visiond.py --model /var/lib/robot/vision/yolo26n.onnx --imgsz 320 --threads 2
```

`--dry-run` still opens the camera and the model, and prints the IPC calls without connecting. `--socket` overrides `/run/robotd.sock`.

`deploy/install.py` copies the sidecar to `/opt/robot/vision/` and installs `easyduck-vision.service`. `deploy/start.sh` does not start that service. Enable it only after the weights, packages, and camera are present; otherwise the unit restarts on the missing model.

```sh
systemctl start easyduck-vision.service
journalctl -u easyduck-vision.service -f
```

Support the robot before starting vision against a live `robotd`. Look, pickup, and stop all request motion or audio.

## Behaviors

A person is greeted once per track, then again only after `--greet-cooldown` seconds (default 8). The head keeps following while the person remains the largest person in frame. Image right is trunk-right, which is negative `y` in `robot.look`. The gaze distance is fixed at 1 m because the detector has no depth.

Pickup is once per track until `--pick-cooldown` seconds (default 6). The box must sit in the lower part of the frame and stay below 12% of the image. A larger forward box is a stop, including a person whose box covers at least 20% of the image in the center-lower region. Other classes stop at 8% of the image in that region.

These thresholds are in `vision/behave.py`. Changing them does not change `robotd`.

## XGO path

The generic controller already consumes `robot.look` and `robot.stop` directly. The installed `ROBOTD_CONTROL_PROFILE=xgoduck` loop consumes the web/gamepad command instead. `robotd` therefore keeps a one-second overlay from those methods, and from `robot.mouth`, and gives the controller a new command revision so the overlay is not dropped. Low-battery latch still refuses the requests. An in-progress pickup is not cancelled by a later stop; the pick policy ignores new velocity until it finishes.

## Checks

From the repository root. The tests use synthetic tensors and a local socket. They do not open a camera, download weights, or move a robot.

```sh
python3 -m unittest discover -s vision/tests -v
```

Compilation and these checks do not establish physical behavior. Confirm the camera view, the voice, and each motion on the robot before leaving it unsupervised.

## Limits

- There is no depth, so gaze and "ahead" are image heuristics.
- COCO does not contain every obstacle. A blank wall is invisible to this detector.
- Stop overrides a stick only while the overlay is fresh. Stopping the sidecar returns velocity to the current web or gamepad command.
- The sidecar is optional. Walking, the web page, and the gamepad work without it.
