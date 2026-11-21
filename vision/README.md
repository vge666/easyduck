# Vision sidecar

**English** | [简体中文](README.zh-CN.md)

## Responsibilities

Optional YOLO detect/track process. It reads the USB camera and submits existing robotd intents for a greeting, a pickup, and an obstacle stop. It does not own the servo bus or the 50 Hz loop.

## File structure

| File / directory | Responsibility |
| --- | --- |
| [visiond.py](visiond.py) | Camera loop, detector selection, and IPC send |
| [detect.py](detect.py) | Ultralytics track/predict, ONNX Runtime decode, letterbox, and IoU tracking |
| [behave.py](behave.py) | Person, small-object, and obstacle decisions |
| [ipc.py](ipc.py) | JSON-RPC client for `/run/robotd.sock` |
| [tests/test_vision.py](tests/test_vision.py) | Synthetic detections, decode, tracking, and socket checks |
| [requirements.lock](requirements.lock) | Extra Python packages; not used by the rest of the robot |
| [../web/usb_camera.py](../web/usb_camera.py) | Existing MJPEG capture used for frames |

## Call and data flow

1. `visiond.py` pulls MJPEG from the UVC camera through `web/usb_camera.py` and decodes it with Pillow or OpenCV.
2. A `.pt` model uses `ultralytics.YOLO.track`. A `.onnx` model uses ONNX Runtime. The package source is [vge666/ultralytics](https://github.com/vge666/ultralytics); this directory does not vendor it.
3. `behave.py` turns boxes into `robot.look`, `robot.sound` (`greet`), `robot.mouth`, `robot.do` (`ground_pick`), and `robot.stop`.
4. `robotd` accepts those methods on its existing socket. The XGO profile overlays a fresh look, mouth, or stop onto the command the 50 Hz loop already reads.

### Direct relationships

- Local dependencies: [web USB capture](../web/usb_camera.py), [robotd IPC](../runtime/robotd/README.md)
- Callers: an operator or `easyduck-vision.service`. Nothing else starts it.

## Boundaries and change notes

- Weights, PyTorch, and the Ultralytics package are optional runtime inputs. Do not commit them.
- Threshold changes belong in `behave.py`. Do not push image logic into the control loop.
- Offline tests do not show that a physical duck looked, picked, or stopped.

## Further reading

- [How to run it](../docs/en/vision.md)
- [Documentation index](../docs/en/README.md)
- [Repository maintenance rules](../AGENTS.md)

From the repository root:

```sh
python3 -m unittest discover -s vision/tests -v
python3 vision/visiond.py --help
```
