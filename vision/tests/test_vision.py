"""Decision, ONNX decode, tracking, and IPC checks. No camera and no weights."""
from __future__ import annotations

import json
import socket
import threading
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from behave import Config, Memory, decide, gaze
from detect import Detection, Tracker, decode_outputs, letterbox
from ipc import IpcError, RobotClient


def box(cls, cx, cy, w, h, conf=0.9, track=1):
    return Detection(cx, cy, w, h, conf, cls, track)


class BehaviorTests(unittest.TestCase):
    def test_person_ahead_greets_once_and_looks_level(self):
        memory = Memory()
        person = box(0, 0.5, 0.5, 0.2, 0.4, track=7)
        first = decide([person], 10.0, memory)
        methods = [item.method for item in first]
        self.assertEqual(methods, ['robot.look', 'robot.sound', 'robot.mouth'])
        self.assertEqual(first[0].params['y'], 0.0)
        self.assertEqual(first[0].params['z'], 0.0)
        self.assertEqual(first[1].params, {'tag': 'greet'})
        self.assertEqual(first[2].params['open'], 0.8)
        second = decide([person], 11.0, memory)
        self.assertEqual([item.method for item in second], ['robot.look'])

    def test_person_on_the_right_looks_to_the_robots_right(self):
        x, y, z = gaze(box(0, 1.0, 0.5, 0.2, 0.2))
        self.assertEqual(x, 1.0)
        self.assertLess(y, 0)
        self.assertEqual(z, 0.0)

    def test_small_ball_requests_pickup_once(self):
        memory = Memory()
        ball = box(32, 0.5, 0.8, 0.1, 0.1, track=3)
        first = decide([ball], 1.0, memory)
        self.assertEqual(first, [type(first[0])('robot.do', {'skill': 'ground_pick'})])
        self.assertEqual(decide([ball], 2.0, memory), [])
        again = decide([ball], 8.0, memory, Config(pick_cooldown_s=6))
        self.assertEqual([item.method for item in again], ['robot.do'])

    def test_large_forward_object_stops_and_suppresses_pickup(self):
        chair = box(56, 0.5, 0.7, 0.4, 0.4, track=2)
        ball = box(32, 0.5, 0.85, 0.1, 0.1, track=4)
        intents = decide([chair, ball], 1.0, Memory())
        self.assertEqual([item.method for item in intents], ['robot.stop'])

    def test_close_person_stops_and_greets(self):
        person = box(0, 0.5, 0.6, 0.5, 0.5, track=9)
        methods = [item.method for item in decide([person], 1.0, Memory())]
        self.assertEqual(methods[0], 'robot.stop')
        self.assertIn('robot.sound', methods)
        self.assertIn('robot.look', methods)

    def test_distant_person_does_not_stop(self):
        person = box(0, 0.5, 0.4, 0.1, 0.2, track=1)
        methods = [item.method for item in decide([person], 1.0, Memory())]
        self.assertNotIn('robot.stop', methods)


class DecodeTests(unittest.TestCase):
    def test_end_to_end_box_maps_back_to_the_image(self):
        rows = np.zeros((1, 2, 6), np.float32)
        rows[0, 0] = (100, 120, 180, 200, 0.9, 0)
        found = decode_outputs([rows], 0.5, 0, 40, 640, 480, 0.35, 0.45)
        self.assertEqual(len(found), 1)
        det = found[0]
        self.assertEqual(det.cls, 0)
        self.assertAlmostEqual(det.cx, 0.4375)
        self.assertAlmostEqual(det.cy, 0.5)
        self.assertAlmostEqual(det.w, 0.25)
        self.assertAlmostEqual(det.h, 160 / 480)

    def test_classic_head_applies_nms(self):
        values = np.zeros((1, 84, 100), np.float32)
        values[0, 0:4, 0] = (160, 160, 40, 40)
        values[0, 4, 0] = 0.9
        values[0, 0:4, 1] = (162, 160, 40, 40)
        values[0, 4, 1] = 0.5
        found = decode_outputs([values], 1.0, 0, 0, 320, 320, 0.35, 0.45)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0].conf, 0.9)
        self.assertEqual(found[0].cls, 0)

    def test_letterbox_pads_the_short_side(self):
        image = np.zeros((480, 640, 3), np.uint8)
        image[0, 0] = (255, 0, 0)
        canvas, scale, left, top = letterbox(image, 320)
        self.assertEqual(canvas.shape, (320, 320, 3))
        self.assertAlmostEqual(scale, 0.5)
        self.assertEqual(left, 0)
        self.assertEqual(top, 40)
        self.assertEqual(tuple(canvas[0, 0]), (114, 114, 114))

    def test_tracker_keeps_an_id_and_drops_a_lost_one(self):
        tracker = Tracker(max_missed=1)
        first = tracker.update([box(0, 0.5, 0.5, 0.2, 0.2, track=None)])
        second = tracker.update([box(0, 0.52, 0.5, 0.2, 0.2, track=None)])
        self.assertEqual(first[0].track, second[0].track)
        tracker.update([])
        tracker.update([])
        third = tracker.update([box(0, 0.1, 0.1, 0.2, 0.2, track=None)])
        self.assertNotEqual(third[0].track, first[0].track)

    def test_unknown_output_is_rejected(self):
        with self.assertRaises(ValueError):
            decode_outputs([np.zeros((1, 3, 3), np.float32)], 1, 0, 0, 8, 8, 0.1, 0.5)


class IpcTests(unittest.TestCase):
    def test_call_uses_jsonrpc_and_reports_refusal(self):
        ready = threading.Event()
        received = []

        def serve(path):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(path)
                listener.listen(1)
                ready.set()
                conn, _ = listener.accept()
                with conn, conn.makefile('rb') as reader:
                    for _ in range(2):
                        line = reader.readline()
                        message = json.loads(line)
                        received.append(message)
                        if message['method'] == 'robot.stop':
                            body = {'jsonrpc': '2.0', 'id': message['id'], 'result': {'accepted': True}}
                        else:
                            body = {'jsonrpc': '2.0', 'id': message['id'],
                                    'result': {'accepted': False, 'reason': 'no policy configured for that skill'}}
                        conn.sendall((json.dumps(body) + '\n').encode())

        path = Path('/tmp') / f'vision-ipc-{threading.get_ident()}.sock'
        path.unlink(missing_ok=True)
        thread = threading.Thread(target=serve, args=(str(path),))
        thread.start()
        self.assertTrue(ready.wait(2))
        client = RobotClient(str(path))
        try:
            self.assertEqual(client.call('robot.stop', {})['accepted'], True)
            with self.assertRaises(IpcError):
                client.call('robot.do', {'skill': 'ground_pick'})
        finally:
            client.close()
            thread.join(2)
            path.unlink(missing_ok=True)
        self.assertEqual(received[0]['jsonrpc'], '2.0')
        self.assertEqual(received[0]['method'], 'robot.stop')
        self.assertEqual(received[1]['params'], {'skill': 'ground_pick'})

    def test_dry_run_does_not_open_a_socket(self):
        client = RobotClient('/tmp/does-not-exist.sock', dry_run=True)
        self.assertTrue(client.call('robot.sound', {'tag': 'greet'})['dry_run'])


if __name__ == '__main__':
    unittest.main()
