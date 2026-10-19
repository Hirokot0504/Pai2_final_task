#!/usr/bin/env python3
"""
電気室 巡回点検ノード(ROS 2 Jazzy)

動作:
  1. 盤の前(1番→2番→3番)へ順に移動して停止する(/odom による位置制御)
  2. 停止中のカメラ画像(/camera)を画像認識AI(CLIP、ゼロショット分類)で
     「正常 / 異常」に判定する
  3. 画素の色解析で、赤い表示灯の位置(左・中央・右)を特定する
  4. 全盤の点検後、点検記録(report.md / results.json)と判定画像を保存する

前提:
  - Gazebo で worlds/inspection_world.sdf を起動済み
  - scripts/bridge.sh でトピックを ROS 2 に接続済み
"""
import datetime
import json
import os
import time

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from PIL import ImageDraw
from rclpy.node import Node
from sensor_msgs.msg import Image
from transformers import CLIPModel, CLIPProcessor

# 盤の名前と x 座標(ワールドファイルと合わせる)
PANELS = [("1番盤", 2.0), ("2番盤", 5.0), ("3番盤", 8.0)]

# カメラはロボット原点から x 方向に -0.151 m の位置にある(SDF の chassis の位置)
CAMERA_OFFSET_X = -0.151427

# CLIP に与える2つの説明文(どちらに近いかで判定する)
MODEL_NAME = "openai/clip-vit-base-patch32"
PROMPTS = [
    "a photo of a green indicator light",
    "a photo of a red indicator light",
]
ABNORMAL_THRESHOLD = 0.5  # 「赤」の確率がこれを超えたら異常と判定

# 表示灯の注目領域(ROI)。画像 320x240 上の各表示灯の中心座標(画素)
# 盤とカメラの位置関係(距離約1.74 m、焦点距離約277画素)から計算した値
LAMP_ROIS = [("左", 120, 64), ("中央", 160, 64), ("右", 200, 64)]
ROI_HALF = 18  # 切り出す正方形の半分の大きさ(画素)


def find_lamps(img):
    """点灯している表示灯(彩度の高い小領域)を探し、(位置名, 中心x, 中心y, 半径) を返す。
    色が赤か緑かは見ない(色の判定は AI に任せる)。見つからなければ固定座標を使う。"""
    mx = img.max(axis=2).astype(int)
    mn = img.min(axis=2).astype(int)
    lit = ((mx - mn) > 80) & (mx > 100)
    ys, xs = np.nonzero(lit)
    if len(xs) < 20:
        return [(n, cx, cy, ROI_HALF) for n, cx, cy in LAMP_ROIS]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    groups, start = [], 0
    for k in range(1, len(xs) + 1):
        if k == len(xs) or xs[k] - xs[k - 1] > 4:  # x 方向に4画素以上の隙間で区切る
            if k - start >= 15:
                groups.append((xs[start:k], ys[start:k]))
            start = k
    lamps = []
    for gx, gy in groups:
        cx, cy = int((gx.min() + gx.max()) / 2), int((gy.min() + gy.max()) / 2)
        half = int(max(gx.max() - gx.min(), gy.max() - gy.min()) / 2) + 3
        lamps.append((cx, cy, max(half, 6)))
    w = img.shape[1]
    named = []
    for cx, cy, half in lamps:
        name = "左" if cx < w * 0.44 else ("右" if cx > w * 0.56 else "中央")
        named.append((name, cx, cy, half))
    return named or [(n, cx, cy, ROI_HALF) for n, cx, cy in LAMP_ROIS]

RESULT_DIR = os.path.expanduser("~/robot_ws/results")

MAX_SPEED = 0.5      # m/s
MIN_SPEED = 0.08     # m/s(目標近くでも止まらないための下限)
POSITION_TOL = 0.03  # m(停止判定の許容誤差)
SETTLE_SEC = 1.5     # 停止後、画像が安定するまで待つ時間


class Inspector(Node):
    def __init__(self):
        super().__init__("inspector")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 10)
        self.create_subscription(Image, "/camera", self.on_image, 10)

        self.x = None
        self.latest = None
        self.warned_encoding = False

        self.get_logger().info(f"画像認識AI({MODEL_NAME})を読み込み中...(初回はダウンロードに数分)")
        self.model = CLIPModel.from_pretrained(MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.get_logger().info("読み込み完了。/odom と /camera を待っています")

        self.idx = 0
        self.state = "WAIT"
        self.stop_time = None
        self.results = []
        self.start_time = datetime.datetime.now()
        os.makedirs(RESULT_DIR, exist_ok=True)

        self.timer = self.create_timer(0.1, self.loop)

    # ---------- 受信 ----------
    def on_odom(self, msg):
        self.x = msg.pose.pose.position.x

    def on_image(self, msg):
        ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(msg.encoding)
        if ch is None:
            if not self.warned_encoding:
                self.get_logger().warn(f"未対応の画像形式です: {msg.encoding}")
                self.warned_encoding = True
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        arr = arr[:, : msg.width * ch].reshape(msg.height, msg.width, ch)
        if msg.encoding.startswith("bgr"):
            arr = arr[:, :, [2, 1, 0]]
        self.latest = arr[:, :, :3].copy()

    # ---------- 制御 ----------
    def publish_speed(self, v):
        t = Twist()
        t.linear.x = float(v)
        self.cmd_pub.publish(t)

    def loop(self):
        if self.state == "WAIT":
            if self.x is not None and self.latest is not None:
                self.get_logger().info("巡回を開始します")
                self.state = "MOVE"
            return

        if self.state == "MOVE":
            name, panel_x = PANELS[self.idx]
            target = panel_x - CAMERA_OFFSET_X
            err = target - self.x
            if abs(err) < POSITION_TOL:
                self.publish_speed(0.0)
                self.get_logger().info(f"{name} の前に到着(x={self.x:.2f})")
                self.state = "SETTLE"
                self.stop_time = time.time()
            else:
                v = max(-MAX_SPEED, min(MAX_SPEED, 0.8 * err))
                if abs(v) < MIN_SPEED:
                    v = MIN_SPEED if err > 0 else -MIN_SPEED
                self.publish_speed(v)

        elif self.state == "SETTLE":
            self.publish_speed(0.0)
            if time.time() - self.stop_time > SETTLE_SEC:
                self.inspect(self.idx)
                self.idx += 1
                if self.idx >= len(PANELS):
                    self.finish()
                    self.state = "DONE"
                else:
                    self.state = "MOVE"

        elif self.state == "DONE":
            self.publish_speed(0.0)

    # ---------- 点検 ----------
    def inspect(self, i):
        name, _ = PANELS[i]
        img = self.latest.copy()
        pil = PILImage.fromarray(img)

        # (1) AI 判定:表示灯ごとに領域を切り出し、CLIP でゼロショット分類
        lamps = find_lamps(img)
        crops = []
        for k, (_, cx, cy, half) in enumerate(lamps):
            box = (cx - half, cy - half, cx + half, cy + half)
            crop = pil.crop(box).resize((224, 224), PILImage.NEAREST)
            crop.save(os.path.join(RESULT_DIR, f"panel_{i + 1}_lamp_{k + 1}.png"))
            crops.append(crop)
        inputs = self.processor(text=PROMPTS, images=crops, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs = self.model(**inputs).logits_per_image.softmax(dim=1)
        p_red = probs[:, 1].tolist()  # 各表示灯が「赤」である確率
        lamp_results = {lamps[k][0]: round(p, 3) for k, p in enumerate(p_red)}
        k_max = int(np.argmax(p_red))
        p_abnormal = p_red[k_max]
        verdict = "異常" if p_abnormal > ABNORMAL_THRESHOLD else "正常"
        ai_position = lamps[k_max][0] if verdict == "異常" else None
        # ROI の枠を判定画像に描く
        draw0 = ImageDraw.Draw(pil)
        for k, (_, cx, cy, half) in enumerate(lamps):
            c = (255, 40, 40) if (verdict == "異常" and k == k_max) else (255, 255, 0)
            draw0.rectangle([cx - half - 1, cy - half - 1, cx + half + 1, cy + half + 1], outline=c)

        # (2) 色解析(AI 判定の照合用):赤い画素の数と位置
        r = img[:, :, 0].astype(int)
        g = img[:, :, 1].astype(int)
        b = img[:, :, 2].astype(int)
        red = (r > 150) & (g < 90) & (b < 90)
        green = (g > 120) & (r < 90) & (b < 90)
        n_red, n_green = int(red.sum()), int(green.sum())
        color_position = None
        if n_red > 20:
            cx = np.nonzero(red)[1].mean() / img.shape[1]
            color_position = "左" if cx < 0.4 else ("右" if cx > 0.6 else "中央")
        agree = ai_position == color_position

        # 判定画像を保存(動画・レポート用)
        draw = ImageDraw.Draw(pil)
        label = f"Panel {i + 1}: {'ABNORMAL' if verdict == '異常' else 'NORMAL'}  p={p_abnormal:.2f}"
        color = (255, 40, 40) if verdict == "異常" else (40, 200, 40)
        draw.rectangle([0, 0, pil.width, 18], fill=(0, 0, 0))
        draw.text((4, 3), label, fill=color)
        img_path = os.path.join(RESULT_DIR, f"panel_{i + 1}.png")
        pil.save(img_path)

        result = {
            "panel": name,
            "verdict": verdict,
            "p_abnormal": round(p_abnormal, 3),
            "lamp_p_red": lamp_results,
            "red_lamp_position": ai_position,
            "color_check_position": color_position,
            "ai_color_agree": agree,
            "red_pixels": n_red,
            "green_pixels": n_green,
            "image": os.path.basename(img_path),
        }
        self.results.append(result)
        probs_txt = " ".join(f"{k}{v:.2f}" for k, v in lamp_results.items())
        self.get_logger().info(
            f"【{name}】AI判定: {verdict} / 各表示灯の赤確率 [{probs_txt}] / "
            f"AIの異常位置: {ai_position or 'なし'} / 色解析: {color_position or 'なし'}"
            f"({'一致' if agree else '不一致'})"
        )

    def finish(self):
        end = datetime.datetime.now()
        abnormal = [r for r in self.results if r["verdict"] == "異常"]

        with open(os.path.join(RESULT_DIR, "results.json"), "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        lines = [
            "# 電気室 巡回点検記録",
            "",
            f"- 点検日時: {self.start_time:%Y-%m-%d %H:%M:%S} 〜 {end:%H:%M:%S}",
            f"- 点検対象: {len(self.results)} 面",
            "- 判定方法: 表示灯ごとの領域切り出し + 画像認識AI(CLIP ゼロショット分類)、色解析で照合",
            "",
            "| 盤 | 判定 | 赤確率(最大) | 異常灯の位置 | 色解析との照合 |",
            "|---|---|---|---|---|",
        ]
        for r in self.results:
            lines.append(
                f"| {r['panel']} | {r['verdict']} | {r['p_abnormal']:.2f} | "
                f"{r['red_lamp_position'] or '-'} | {'一致' if r['ai_color_agree'] else '不一致'} |"
            )
        lines.append("")
        if abnormal:
            for r in abnormal:
                pos = f"{r['red_lamp_position']}の" if r["red_lamp_position"] else ""
                lines.append(f"- {r['panel']}:{pos}表示灯が赤点灯。現地確認と原因調査を要する。")
        else:
            lines.append("- 全盤正常。異常なし。")
        text = "\n".join(lines) + "\n"

        with open(os.path.join(RESULT_DIR, "report.md"), "w", encoding="utf-8") as f:
            f.write(text)
        self.get_logger().info("点検完了。記録を保存しました: " + RESULT_DIR)
        print("\n" + text)


def main():
    rclpy.init()
    node = Inspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_speed(0.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
