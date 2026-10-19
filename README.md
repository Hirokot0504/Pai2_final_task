## 電気室 巡回点検ロボット(ROS 2 + Gazebo + 画像認識AI)

Gazebo上の電気室で、カメラ付き移動ロボットが盤を巡回し、表示灯の状態を画像認識AI(CLIP)で判定して点検記録を自動作成するシミュレーション。

## 概要

- 動力盤3面(1番〜3番)の全面を順に巡回し、各盤の前で停止して表示灯を撮影
- 点灯している表示灯を画像から自動検出して切り出し、CLIP(ゼロショット分類)で「緑/赤」を判定
- 色解析の結果と照合し、異常の有無と異常灯の位置(左・中央・右)を特定
- 点検後、点検記録(`report.md`)・判定結果(`results.json`)・判定画像を出力

## 環境

- macOS(Apple Silicon)+ Docker
- イメージ: `ghcr.io/tiryoh/ros2-desktop-vnc:jazzy`(Ubuntu 24.04 / ROS 2 Jazzy)
- Gazebo Harmonic(`ros-jazzy-ros-gz`)
- Python: torch(CPU)、transformers、pillow、numpy<2
- AIモデル: `openai/clip-vit-base-patch32`(学習なし、推論のみ)

## 構成

```
worlds/inspection_world.sdf   電気室内・盤3面・カメラ付きロボット
scripts/bridge.sh             Gazebo と ROS 2 のトピック接続
scripts/inspector_node.py     巡回・AI判定・点検記録作成ノード
results/                      実行結果(点検記録・判定画像)
```

## 実行方法

```
# 準備(初回のみ)
sudo apt update
sudo apt install -y ros-jazzy-ros-gz python3-pip
pip3 install --break-system-packages "numpy<2" torch transformers pillow

# ターミナル1:シミュレーション
gz sim -r worlds/inspection_world.sdf
# ターミナル2:トピック接続
bash scripts/bridge.sh
# ターミナル3:巡回点検
python3 scripts/inspector_node.py
```


