# 视觉旁路

[English](README.md) | **简体中文**

## 职责

可选的 YOLO 检测／跟踪进程。它读取 USB 摄像头，并把问候、捡物和遇障停止提交为 robotd 已有的意图。它不占用舵机总线，也不进入 50 Hz 循环。

## 文件结构

| 文件／目录 | 职责 |
| --- | --- |
| [visiond.py](visiond.py) | 摄像头循环、检测器选择和 IPC 发送 |
| [detect.py](detect.py) | Ultralytics track/predict、ONNX Runtime 解码、letterbox 和 IoU 跟踪 |
| [behave.py](behave.py) | 人、小物体和障碍的判定 |
| [ipc.py](ipc.py) | `/run/robotd.sock` 的 JSON-RPC 客户端 |
| [tests/test_vision.py](tests/test_vision.py) | 合成检测、解码、跟踪和 socket 检查 |
| [requirements.lock](requirements.lock) | 额外 Python 包；机器人其余部分不使用 |
| [../web/usb_camera.py](../web/usb_camera.py) | 采集画面时复用的现有 MJPEG 捕获 |

## 调用与数据流

1. `visiond.py` 经 `web/usb_camera.py` 读取 UVC 摄像头的 MJPEG，并用 Pillow 或 OpenCV 解码。
2. `.pt` 模型使用 `ultralytics.YOLO.track`。`.onnx` 模型使用 ONNX Runtime。包的来源是 [vge666/ultralytics](https://github.com/vge666/ultralytics)；本目录不内置该包。
3. `behave.py` 把检测框变成 `robot.look`、`robot.sound`（`greet`）、`robot.mouth`、`robot.do`（`ground_pick`）和 `robot.stop`。
4. `robotd` 在现有 socket 上接受这些方法。XGO 配置把仍然新鲜的转向、张嘴或停止叠到 50 Hz 循环已经在读的命令上。

### 直接关系

- 本地依赖：[网页 USB 采集](../web/usb_camera.py)、[robotd IPC](../runtime/robotd/README.zh-CN.md)
- 调用方：操作者或 `easyduck-vision.service`。没有其他程序启动它。

## 边界与修改注意

- 权重、PyTorch 和 Ultralytics 包是可选的运行输入。不要把它们提交进仓库。
- 阈值修改放在 `behave.py`。不要把图像逻辑放进控制循环。
- 离线测试不能说明实物鸭子已经完成转向、捡物或停止。

## 相关说明

- [如何运行](../docs/zh-CN/vision.md)
- [文档索引](../docs/zh-CN/README.md)
- [仓库维护规则](../AGENTS.md)

在仓库根目录执行：

```sh
python3 -m unittest discover -s vision/tests -v
python3 vision/visiond.py --help
```
