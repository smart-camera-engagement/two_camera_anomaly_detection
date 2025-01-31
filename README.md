# ADI_two_camera_anomaly_detection

- [ADI\_two\_camera\_anomaly\_detection](#adi_two_camera_anomaly_detection)
  - [About](#about)
  - [Hardware Spec](#hardware-spec)
  - [Prepare](#prepare)
  - [How to start](#how-to-start)
  - [Data Flow](#data-flow)
  - [Time Sequence](#time-sequence)
  - [Reference](#reference)

## About
This documents describe the demo setup of Analog Device two camera pill anormly detection demo. The two camera are:
1. Cam1: Pill detection: Detection pill, send bounding box to Cam 2. Waiting for Cam 2 send back anormly detection result for display
2. Cam2: Get boundary box from Cam1, run anormly detection, and send anormly detection result to Cam1.

## Hardware Spec

1. Raspberry Pi 5
2. Two Rapsberry Pi Ai Camera - Those two camera mount side by side, distance is 40mm
<p align="center">
<image src="./images/two_camera_setup.png" alt="two_camera_setup" width="500">
</p>
3. LED Strip(Optional)

## Prepare

1. Clone [picamera2](https://github.com/raspberrypi/picamera2)
1. Create a folder named "demo" in picamera2/example/ADI
2. Create a folder named "Detection" in picamera2/examples/ADI/Models/, copy your pill detection ai model(network.rpk) and labels(labels.txt) into this folder. 
3. Create a folder named "Anomaly" in picamera2/examples/ADI/Models/, copy your pill anomaly detection ai model(network.rpk) into this folder.
4. Put this repo into folder picamera2/example/ADI
5. Install [SORT](https://github.com/abewley/sort)
<p align="left">
<image src="./images/folder_struct.png" alt="folder_struct" width="300">
</p>

## How to start
```
$ cd xxx/picamera2/examples/ADI
$ python3 main.py
```
## Data Flow
<p align="center">
<image src="./images/data_flow.png" alt="data_flow" width="700">
</p>

## Time Sequence
<p align="center">
<image src="./images/time_sequence.png" alt="data_flow" width="700">
</p>

## Reference
- [Raspberry Pi Ai Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html)
- [picamera2](https://github.com/raspberrypi/picamera2)
- [libcamera](https://github.com/raspberrypi/libcamera)
- [imx500](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera)