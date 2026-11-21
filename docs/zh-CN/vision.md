# YOLO 视觉

[English](../en/vision.md) | **简体中文**

可选的摄像头路径。Python 旁路运行 YOLO，并发送现有的 `robot.look`、`robot.sound`、`robot.mouth`、`robot.do` 和 `robot.stop`。它不进入 50 Hz 循环，不打开舵机总线，也不替换 `robotd`。

## 作用

| 摄像头中看到 | 请求 |
| --- | --- |
| 人 | `robot.look` 转向对方，`robot.sound` 标签 `greet`，以及 `robot.mouth` |
| 画面下方的小型 COCO 物体 | `robot.do` 动作 `ground_pick` |
| 画面前方的大物体，或占满该区域的人 | `robot.stop` |

`greet` 是现有的醒来叫声。嘴部命令与网页控制已经使用的 0～1 张合幅度相同。捡物是现有的地面捡物策略，不会对准检测框。停止把行走速度清零，策略保持启用。

小型类别是 COCO 编号 32、39、41、46、47、49、67 和 73（球、瓶子、杯子、香蕉、苹果、橙子、手机、书）。小物体不会同时被当成障碍。障碍还在前方时不发送捡物。停止期间仍可以问候。

旁路不接管网页或手柄的控制权。在已安装的 XGO 配置上，`robotd` 把仍然新鲜的转向、张嘴或停止叠到当前 XGO 命令上，大约保持 1 秒，然后回到摇杆或网页命令。摄像头仍支持该动作时，旁路会重复发送。

## 运行条件

- `robotd` 已在运行。捡物开始前需要已启用行走策略。没有捡物模型时，请求会带原因被拒绝。
- `greet` 需要现有音效库。没有音效库时 `robot.sound` 被拒绝，嘴部命令仍可发送。
- 提供 MJPEG 的 USB UVC 摄像头。采集复用 `web/usb_camera.py`。
- `vision/requirements.lock` 中的额外 Python 包。`robotd`、网页控制和标定都不需要它们。
- YOLO 权重。本仓库不包含权重。

泰山派是 ARM64，`robotd` 内部已经使用 ONNX Runtime。板端不需要完整的 PyTorch。使用 ONNX 导出和 Python 的 `onnxruntime` 包；它与 `robotd` 加载的原生库不是同一个包。优先使用 1.23 或更新的 ONNX Runtime。

`.pt` 文件的检测／跟踪走 Ultralytics 包。来源是 [vge666/ultralytics](https://github.com/vge666/ultralytics)。上游 Ultralytics 代码和发布的 YOLO 权重在没有另行取得 Ultralytics 许可时为 AGPL-3.0。它们不在本仓库中，也不改授为 Apache-2.0。

## 权重

在可以安装 Ultralytics 和 PyTorch 的开发机上，于 `yolo` 命令所在的任意目录执行：

```sh
pip install ultralytics
yolo export model=yolo26n.pt format=onnx imgsz=320
```

`yolo26n` 是小型 COCO 检测模型。`imgsz=320` 是板端默认值，便于 RK3566 把时间留给控制循环；640 是模型训练尺寸，更重。把生成的 `yolo26n.onnx` 拷到板上。不要把该文件提交进仓库。

```sh
sudo install -d /var/lib/robot/vision
sudo install -m 0644 yolo26n.onnx /var/lib/robot/vision/yolo26n.onnx
```

`--backend auto` 对 `.pt` 选择 Ultralytics，对 `.onnx` 选择 ONNX Runtime。`--backend ultralytics` 调用 `YOLO.track`（ByteTrack）。跟踪无法启动时，旁路改用 `predict` 和自带的 IoU 跟踪。ONNX 路径始终使用该 IoU 跟踪。两条路径都只返回真实检测框，没有框时不返回框。

## 运行

先在板端安装额外的包，再在仓库根目录启动旁路。`--threads 2` 把其余核心留给 `robotd`。

```sh
python3 -m pip install --user numpy pillow 'onnxruntime>=1.23'
python3 vision/visiond.py --model /var/lib/robot/vision/yolo26n.onnx --imgsz 320 --threads 2
```

`--dry-run` 仍会打开摄像头和模型，并打印 IPC 调用，但不连接。`--socket` 可替换 `/run/robotd.sock`。

`deploy/install.py` 把旁路复制到 `/opt/robot/vision/`，并安装 `easyduck-vision.service`。`deploy/start.sh` 不启动该服务。权重、依赖包和摄像头都就绪后再启用；否则单元会因为找不到模型而反复重启。

```sh
systemctl start easyduck-vision.service
journalctl -u easyduck-vision.service -f
```

对着正在运行的 `robotd` 启动视觉前，先扶稳机器人。转向、捡物和停止都会请求动作或声音。

## 行为

每个跟踪目标问候一次，之后只在 `--greet-cooldown` 秒后再次问候（默认 8 秒）。人还是画面中最大的人时，头部继续跟随。图像右侧是躯干右侧，在 `robot.look` 里是负的 `y`。检测器没有深度，注视距离固定为 1 米。

每个跟踪目标在 `--pick-cooldown` 秒内只捡一次（默认 6 秒）。框必须位于画面下部，并且小于画面的 12%。更大的前方框会触发停止，包括画面中下部、面积至少 20% 的人。其他类别在该区域达到画面的 8% 即停止。

这些阈值在 `vision/behave.py`。修改它们不会修改 `robotd`。

## XGO 路径

通用控制器直接使用 `robot.look` 和 `robot.stop`。已安装的 `ROBOTD_CONTROL_PROFILE=xgoduck` 循环使用的是网页／手柄命令。因此 `robotd` 为这些方法以及 `robot.mouth` 保留大约一秒的叠加，并给控制器一个新的命令版本号，避免叠加被丢掉。低电量锁定仍然拒绝这些请求。正在进行的捡物不会被随后的停止取消；捡物策略在结束前不采用新的速度。

## 检查

在仓库根目录执行。测试使用合成张量和本机 socket，不打开摄像头，不下载权重，也不让机器人运动。

```sh
python3 -m unittest discover -s vision/tests -v
```

编译和这些检查不能证明实物行为。让机器人独自运行前，先在实物上确认画面、声音和每个动作。

## 限制

- 没有深度，注视和“前方”都是图像上的判断。
- COCO 不包含所有障碍。一面空墙对这个检测器是不可见的。
- 只有叠加仍然新鲜时，停止才会压过摇杆。旁路退出后，速度回到当前网页或手柄命令。
- 旁路是可选的。没有它，行走、网页和手柄仍然工作。
