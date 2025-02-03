"""
Description   : This script designed to perform simultaneous object detection and anomaly detection using multiprocessing
Author        : Fang Du
Email         : fang.du@sony.com
Date Created  : 2025-01-30
Date Modified : 2025-02-02
Version       : 1.0
Python Version: 3.11
License       : © 2025 - Sony Semiconductor Solution America
History       :
              : 1.0 - 2025-01-30 Create Script
              : 1.1 - 2025-02-02 Add global variables
"""

import argparse
from multiprocessing import Process, Queue
from imx500_object_detection_SORT import detection_process
from imx500_anomaly_detection import anomaly_process

if __name__ == "__main__":

    CAMERA_DISTANCE_MM = 40  # Physical distance between cameras in mm
    CAMERA_DISTANCE_PIXELS = 210  # Distance in pixels
    PIXELS_PER_MM = CAMERA_DISTANCE_PIXELS / CAMERA_DISTANCE_MM
    DETECTION_REGION = [20, 70, 600, 410]

    bbox_queue = Queue(maxsize=50)  # Queue for passing bounding boxes
    results_queue = Queue()  # Queue for receiving classification results

    pill_detection_args = argparse.Namespace(
        model="./Models/Detection/network.rpk", 
        labels="./Models/Detection/labels.txt",
        camera_index=0,
        fps=20,
        max_disappeared=20,
        iou=0.65,
        threshold=0.5,
        max_detections=10,
        pixels_per_mm=PIXELS_PER_MM,
        detection_region=DETECTION_REGION
    )

    anomaly_detection_args = argparse.Namespace(
        model="./Models/Anomaly/network.rpk",
        camera_index=1,
        fps=20,
        image_threshold=0.45,
        pixel_threshold=0.30,
        constant_offset_in_pixel=CAMERA_DISTANCE_PIXELS,
        roi_box_size=128,
        pixels_per_mm=PIXELS_PER_MM
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
