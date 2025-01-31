"""
Description   : This script designed to perform simultaneous object detection and anomaly detection using multiprocessing
Author        : Fang Du
Email         : fang.du@sony.com
Date Created  : 2025-01-30
Date Modified : 2025-01-30
Version       : 1.0
Python Version: 3.11
License       : © 2025 - Sony Semiconductor Solution America
History       :
              : 1.0 - 2025-1-30 Create Script
"""

import argparse
from multiprocessing import Process, Queue
from imx500_object_detection_SORT import detection_process
from imx500_anomaly_detection import anomaly_process

if __name__ == "__main__":
    bbox_queue = Queue(maxsize=50)  # Queue for bounding boxes
    results_queue = Queue()  # Queue for anomaly results

    pill_detection_args = argparse.Namespace(
        model="/home/pi/picamera2/examples/ADI/Models/Detection/network.rpk", # object detection model path
        labels="/home/pi/picamera2/examples/ADI/Models/Detection/labels.txt", # object detection label path
        camera_index=0,
        fps=20,
        max_disappeared=20,
        iou=0.65,
        threshold=0.3,
        max_detections=10,
        pixels_per_mm=190/40
    )

    anomaly_detection_args = argparse.Namespace(
        model="/home/pi/picamera2/examples/ADI/Models/Anomaly/network.rpk",
        camera_index=1,
        fps=20,
        image_threshold=0.40,
        pixel_threshold=0.30,
        constant_offset_in_pixel=190 # camera 0 and camera 1 has a 40 mm distance = 190 pixels
    )

    pill_detection_proc = Process(target=detection_process, args=(bbox_queue, results_queue, pill_detection_args))
    anomaly_detection_proc = Process(target=anomaly_process, args=(bbox_queue, results_queue, anomaly_detection_args))

    pill_detection_proc.start()
    anomaly_detection_proc.start()

    try:
        pill_detection_proc.join()
        anomaly_detection_proc.join()
    except KeyboardInterrupt:
        print("Stopping processes...")
        pill_detection_proc.terminate()
        anomaly_detection_proc.terminate()