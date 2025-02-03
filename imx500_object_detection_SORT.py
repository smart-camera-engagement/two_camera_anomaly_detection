"""
Description   : This script designed to perform object detection + tracking
Author        : Fang Du
Email         : fang.du@sony.com
Date Created  : 2025-01-30
Date Modified : 2025-01-30
Version       : 1.1
Python Version: 3.11
License       : © 2025 - Sony Semiconductor Solution America
History       :
              : 1.0 - 2025-01-30 Create Script
              : 1.1 - 2025-02-02 Add detection region

"""

import time
import cv2
import numpy as np
from sort import Sort
from functools import lru_cache
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                    postprocess_nanodet_detection)

@dataclass
class Detection:
    category: int
    conf: float
    box: List[int]
    tracking_id: Optional[int] = None

    @property
    def bbox_for_sort(self) -> List[float]:
        """Convert box format for SORT tracker."""
        x, y, w, h = self.box
        return [x, y, x + w, y + h]

class IMX500Detector:
    def __init__(self, args):
        self.camera_path = self._select_camera(args.camera_index)
        self.imx500 = IMX500(network_file=args.model, camera_id=self.camera_path)
        self.intrinsics = self.imx500.network_intrinsics
        if not self.intrinsics:
            self.intrinsics = NetworkIntrinsics()
            self.intrinsics.task = "object detection"

        self.tracker = Sort(
            max_age=args.max_disappeared,
            min_hits=3,
            iou_threshold=args.iou
        )
        
        self._configure_intrinsics(args)
        self._setup_camera(args)
        self.last_results = None
        
        # Track processed IDs
        self.processed_ids: Set[int] = set()
        self.anomaly_results = {}
        self.previous_positions = {}  # Store previous positions
        self.pixels_per_mm = args.pixels_per_mm  # Conversion factor

        # Conveyor belt speed
        self.prev_time = 0
        self.conveyor_speed = 0.0    # Current estimated conveyor speed
        self.speed_history = []
        self.history_window = 10     # Number of frames to average speed over

        self.detection_region = args.detection_region

    def estimate_conveyor_speed(self, detections: List[Detection], current_time: float) -> None:
        dt = current_time - self.prev_time
        speeds = []
        for det in detections:
            if det.tracking_id is None:
                continue
            current_pos = self._get_center(det.box)
            if det.tracking_id in self.previous_positions:
                prev = self.previous_positions[det.tracking_id]
                prev_pos = prev['position']
                dy = current_pos[1] - prev_pos[1]
                displacement_mm = dy / self.pixels_per_mm
                speed = displacement_mm / dt
                if abs(speed) < 500:
                    speeds.append(speed)
            self.previous_positions[det.tracking_id] = {
                'position': current_pos,
                'time': current_time
            }
        if speeds:
            current_speed = np.median(speeds)
            self.speed_history.append(current_speed)
            if len(self.speed_history) > self.history_window:
                self.speed_history.pop(0)
            self.conveyor_speed = np.mean(self.speed_history)
        self.prev_time = current_time

    def _get_center(self, box: List[int]) -> Tuple[float, float]:
        x, y, w, h = box
        return (x + w/2, y + h/2)
    
    def _is_point_in_region(self, point: Tuple[float, float], region: List[int]) -> bool:
        x, y = point
        rx, ry, rw, rh = region
        return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)
        
    def cleanup_tracking(self) -> None:
        """Remove tracking data for objects that are no longer visible."""
        if not self.last_results:
            return

        current_ids = {det.tracking_id for det in self.last_results}
        ids_to_remove = set(self.previous_positions.keys()) - current_ids
        for tracking_id in ids_to_remove:
            self.previous_positions.pop(tracking_id, None)

    def cleanup_results(self) -> None:
        if not self.last_results:
            return 
        
        current_tracking_ids = {det.tracking_id for det in self.last_results}

        results_to_remove = [
            tracking_id for tracking_id in self.anomaly_results.key()
            if tracking_id not in current_tracking_ids
        ]

        for tracking_id in results_to_remove:
            del self.anomaly_results[tracking_id]

    def _select_camera(self, camera_index: int) -> str:
        cameras = [
            "/base/axi/pcie@120000/rp1/i2c@88000/imx500@1a",
            "/base/axi/pcie@120000/rp1/i2c@80000/imx500@1a",
        ]
        if camera_index < 0 or camera_index >= len(cameras):
            raise ValueError(f"Invalid camera index: {camera_index}. Available cameras: {len(cameras)}")
        return cameras[camera_index]

    def _configure_intrinsics(self, args) -> None:
        for key, value in vars(args).items():
            if key == 'labels' and value is not None:
                with open(value, 'r') as f:
                    self.intrinsics.labels = f.read().splitlines()
            elif hasattr(self.intrinsics, key) and value is not None:
                setattr(self.intrinsics, key, value)

        # Set defaults if needed
        if self.intrinsics.labels is None:
            with open("assets/coco_labels.txt", "r") as f:
                self.intrinsics.labels = f.read().splitlines()
        self.intrinsics.update_with_defaults()

    def _setup_camera(self, args) -> None:
        self.picam2 = Picamera2(self.imx500.camera_num)
        config = self.picam2.create_preview_configuration(
            controls={"FrameRate": self.intrinsics.inference_rate},
            buffer_count=12
        )
        self.imx500.show_network_fw_progress_bar()
        self.picam2.start(config, show_preview=True)
        if self.intrinsics.preserve_aspect_ratio:
            self.imx500.set_auto_aspect_ratio()

    @lru_cache
    def get_labels(self) -> List[str]:
        labels = self.intrinsics.labels
        if self.intrinsics.ignore_dash_labels:
            labels = [label for label in labels if label and label != "-"]
        return labels

    def process_standard_output(self, np_outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        input_w, input_h = self.imx500.get_input_size()
        boxes, scores, classes = np_outputs[0][0], np_outputs[2][0], np_outputs[1][0]
        if self.intrinsics.bbox_normalization:
            boxes = boxes / input_h

        if self.intrinsics.bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
            
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = list(zip(*boxes))
        return boxes, scores, classes

    def parse_detections(self, metadata, iou: float,
                        max_detections: int, threshold: float) -> List[Detection]:
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if np_outputs is None:
            return []
        boxes, scores, classes = self.process_standard_output(np_outputs)
        detections = [
            Detection(int(category), score, 
                    self.imx500.convert_inference_coords(box, metadata, self.picam2))
            for box, score, category in zip(boxes, scores, classes)
            if score > threshold 
            and self._is_point_in_region(
                self._get_center(
                    self.imx500.convert_inference_coords(box, metadata, self.picam2)
                ),
                self.detection_region
            )
        ]
        # Update tracking
        if not detections:
            tracked_objects = self.tracker.update(np.empty((0, 4)))
            return []

        tracked_objects = self.tracker.update(
            np.array([d.bbox_for_sort for d in detections])
        )
        tracked_detections = []
        for detection, track in zip(detections[:len(tracked_objects)], tracked_objects):
            x_min, y_min, x_max, y_max, track_id = track
            detection.box = [
                int(x_min), int(y_min),
                int(x_max - x_min), int(y_max - y_min)
            ]
            detection.tracking_id = int(track_id)
            tracked_detections.append(detection)
        
        current_time = time.time()
        self.estimate_conveyor_speed(tracked_detections, current_time)

        # Cleanup old tracking data
        self.cleanup_tracking()

        return tracked_detections

    def draw_detections(self, request, results_queue) -> None:
        if self.last_results is None:
            return

        # Get all available classification results
        while not results_queue.empty():
            result = results_queue.get()
            self.anomaly_results[result["id"]] = result["is_anomaly"]

        labels = self.get_labels()
        with MappedArray(request, "main") as m:
            # draw detection region
            b_x, b_y, b_w, b_h = self.detection_region
            color = (255, 255, 0)  # Yellow
            cv2.putText(m.array, "Detection Region", (b_x + 5, b_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y),
                        (b_x + b_w, b_y + b_h), (255, 255, 0, 0))
            
            if self.conveyor_speed and self.conveyor_speed > 1:
                speed_text = f"Conveyor Speed: {abs(self.conveyor_speed):.2f} mm/s"
                cv2.putText(m.array, speed_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            for detection in self.last_results:
                x, y, w, h = detection.box
                label = f"ID:{detection.tracking_id}-{labels[detection.category]}({detection.conf:.2f})"

                # Add classification result if available
                if detection.tracking_id in self.anomaly_results:
                    if not self.anomaly_results[detection.tracking_id]:
                        label += f" - N"
                    else:
                        label += f" - A"

                # Draw text background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                text_x = x + 5
                text_y = y - 10

                overlay = m.array.copy()
                cv2.rectangle(overlay,
                            (text_x, text_y - text_height),
                            (text_x + text_width, text_y + baseline),
                            (255, 255, 255),
                            cv2.FILLED)

                cv2.addWeighted(overlay, 0.3, m.array, 0.7, 0, m.array)

                # Draw text and box
                cv2.putText(m.array, label, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), 2)
            # Draw ROI if needed
            if self.intrinsics.preserve_aspect_ratio:
                b_x, b_y, b_w, b_h = self.imx500.get_roi_scaled(request)
                cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.rectangle(m.array, (b_x, b_y),
                            (b_x + b_w, b_y + b_h), (255, 0, 0, 0))

    def update_bbox_queue(self, bbox_queue) -> None:
        if not self.last_results:
            return
        
        for detection in self.last_results:
            if (detection.tracking_id not in self.processed_ids 
                and not bbox_queue.full()):
                x, y, w, h = detection.box
                bbox_data = {
                    "time": time.time(),
                    "id": detection.tracking_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "speed": int(self.conveyor_speed),
                }
                bbox_queue.put(bbox_data)
                self.processed_ids.add(detection.tracking_id)
                print(f"Added to bbox_queue: {bbox_data}")

def detection_process(bbox_queue, results_queue, args):
    detector = IMX500Detector(args)
    detector.picam2.pre_callback = lambda req: detector.draw_detections(req, results_queue)
    while True:
        metadata = detector.picam2.capture_metadata()
        detector.last_results = detector.parse_detections(
            metadata,
            args.iou,
            args.max_detections,
            args.threshold
        )
        detector.update_bbox_queue(bbox_queue)