# Dual-camera Pill Anomaly Detection

## About
This document describes the demo setup for a dual-camera pill anomaly detection system. The two cameras function as follows:
1. Cam1 (Pill Detection): 
   - Detects pills and sends bounding box coordinates to Cam2.
   - Waits for Cam2 to send back anomaly detection results for display.
2. Cam2 (Anomaly Detection): 
    - Receives the bounding box coordinates from Cam1.
    - Performs anomaly detection on the detected pill.
    - Sends the anomaly detection results back to Cam1.

## Hardware Spec
1. Raspberry Pi 5
2. Two Rapsberry Pi Ai Camera - Those two camera mount side by side, distance is 40mm. The distance between camera to pill is ~20cm.
<p align="center">
<image src="./images/two_camera_setup.png" alt="two_camera_setup" width="500">
</p>
3. LED Strip(Optional)

## Prepare
1. Make the [setup.sh](setup.sh) executable.
   ```
   chmod +x setup.sh
   ```
2. Run it with sudo:
   ```
   sudo ./setup.sh
   ```
3. After it finished, the folder structure as below.
   ```
   ├── images
   ├── Models
   │     ├── Anomaly
   |     |     └── network.rpk
   |     └── Detection
   |           |── labels.txt
   |           └── network.rpk
   ├── imx500_anomaly_detection.py
   ├── imx500_object_detection_SORT.py
   ├── main.py
   ├── sort.py
   ├── setup.sh
   └── README.md
   ```
4. Reboot the raspberry pi.
   ```
   sudo reboot
   ```

## How to run

1. Update the model path and other parameters in [main.py](main.py) as needed.
2. Run the script using the following command:
    ```
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
- [BrainBuilder](https://support.neurala.com/docs/using-brain-builder)