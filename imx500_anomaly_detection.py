"""
Description   : This script designed to perform pill anormly detection
Author        : Fang Du
Email         : fang.du@sony.com
Date Created  : 2025-01-30
Date Modified : 2025-02-02
Version       : 1.2
Python Version: 3.11
License       : © 2025 - Sony Semiconductor Solution America
History       :
              : 1.0 - 2025-01-30 Create Script
              : 1.1 - 2025-01-31 Modified to three-frame cycle
              : 1.2 - 2025-02-02 Add frame number, calculate ROI based on speed, improve three-frame cycle
"""

import time
import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices import IMX500

@dataclass
class AnomalyResult:
    bbox_id: int
    image_score: float
    pixel_scores: np.ndarray
    mask: np.ndarray

class IMX500AnomalyDetector:
    def __init__(self, args):
        self.camera_path = self._select_camera(args.camera_index)
        self.imx500 = IMX500(network_file=args.model, camera_id=self.camera_path)
        
        # Store thresholds from arguments
        self.image_threshold = args.image_threshold
        self.pixel_threshold = args.pixel_threshold
        self.constant_offset = args.constant_offset_in_pixel
        self.roi_box_size = args.roi_box_size
        self.pixels_per_mm = args.pixels_per_mm
        
        # Initialize Picamera
        self.picam2 = Picamera2(self.imx500.camera_num)
        self._setup_camera(args)

        self.frame_state = 0
        self.current_bbox_id = None
        self.processed_bbox_ids = set()
        self.detection_time = None

        self.frame_count = 0

    def _select_camera(self, camera_index: int) -> str:
        cameras = [
            "/base/axi/pcie@120000/rp1/i2c@88000/imx500@1a",
            "/base/axi/pcie@120000/rp1/i2c@80000/imx500@1a",
        ]
        if camera_index < 0 or camera_index >= len(cameras):
            raise ValueError(f"Invalid camera index: {camera_index}. Available cameras: {len(cameras)}")
        return cameras[camera_index]

    def _setup_camera(self, args) -> None:
        config = self.picam2.create_preview_configuration(
            controls={"FrameRate": args.fps},
            buffer_count=12
        )
        self.imx500.show_network_fw_progress_bar()
        self.picam2.start(config, show_preview=True)

    def scale_bbox_to_detection(self, detection_bbox: Tuple[int, int, int, int, float, int],
                              scale_x: float, scale_y: float) -> Tuple[int, int, int, int]:
        x, y, w, h, detection_time, speed = detection_bbox

        y_displacement = 0
        if detection_time is not None and speed != 0:
            current_time = time.time()
            time_diff = current_time - detection_time
            # Convert speed from mm/s to pixels/s and calculate displacement
            speed_pixels = speed * self.pixels_per_mm
            y_displacement = speed_pixels * time_diff
        center_x = x + w / 2
        center_y = y + h / 2 + y_displacement
        new_x = center_x - self.roi_box_size / 2
        new_y = center_y - self.roi_box_size / 2
        scaled_w = int(self.roi_box_size * scale_x) 
        scaled_h = int(self.roi_box_size * scale_y) 
        scaled_x = int((new_x - self.constant_offset) * scale_x)
        scaled_y = int(new_y * scale_x)
        scaled_x = max(0, scaled_x)
        scaled_y = max(0, scaled_y)
        return (scaled_x, scaled_y, scaled_w, scaled_h)

    def process_anomaly_results(self, request: CompletedRequest) -> Optional[AnomalyResult]:
        raw_outputs = self.imx500.get_outputs(request.get_metadata())
        if raw_outputs is None:
            print("Warning: Model output is None.")
            return None
        np_output = np.squeeze(raw_outputs)
        image_score = np_output[0, 0, 1]
        pixel_scores = np_output[:, :, 0]
        mask = np.zeros(pixel_scores.shape + (4,), dtype=np.uint8)
        mask[pixel_scores >= self.pixel_threshold, 0] = 255  # Red channel
        mask[pixel_scores >= self.pixel_threshold, 3] = 150  # Alpha channel
        return AnomalyResult(
            bbox_id=self.current_bbox_id,
            image_score=image_score,
            pixel_scores=pixel_scores,
            mask=mask
        )

    def draw_anomaly_results(self, request: CompletedRequest, 
                           results: AnomalyResult,
                           stream: str = "main") -> None:
        with MappedArray(request, stream) as m:
            # draw ROI
            b_x, b_y, b_w, b_h = self.imx500.get_roi_scaled(request)
            color = (255, 255, 0)  # Yellow
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y),
                        (b_x + b_w, b_y + b_h), (255, 255, 0, 0))

            result_text = "Anomaly" if results.image_score > self.image_threshold else "Normal"
            score_text = f"Score: {results.image_score:.3f}"
            bbox_text = f"BBox ID: {results.bbox_id}"
            
            cv2.putText(m.array, result_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(m.array, score_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(m.array, bbox_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Mask
            resized_mask = cv2.resize(results.mask, (b_w, b_h), 
                                    interpolation=cv2.INTER_NEAREST)
            roi = m.array[b_y:b_y+b_h, b_x:b_x+b_w]
            overlay = np.zeros_like(roi)
            overlay[resized_mask[:, :, 3] > 0] = [255, 255, 0, 150]  # Red with alpha
            cv2.addWeighted(overlay, 0.5, roi, 1.0, 0, roi)
            m.array[b_y:b_y+b_h, b_x:b_x+b_w] = roi

    def draw_roi(self, request: CompletedRequest, stream: str = "main") -> None:
        with MappedArray(request, stream) as m:
            # draw ROI
            b_x, b_y, b_w, b_h = self.imx500.get_roi_scaled(request)
            color = (255, 255, 0)  # Yellow
            cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(m.array, (b_x, b_y),
                        (b_x + b_w, b_y + b_h), (255, 255, 0, 0))

    def draw_frame_number(self, request: CompletedRequest, stream: str = "main") -> None:
        with MappedArray(request, stream) as m:
            frame_text = f"Frame: {self.frame_count}"
            cv2.putText(m.array, frame_text, (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)  # Green color
    
    def process_frame(self, request: CompletedRequest, bbox_queue, results_queue) -> None:

        self.draw_frame_number(request)
        self.frame_count += 1
        try:
            if self.frame_state == 0:  # setting ROI
                # testing code
                # bbox = {
                #     'x': 400,
                #     'y': 250,
                #     'w': 60,
                #     'h': 60,
                #     'id': 1,
                #     'time': time.time(),
                #     'speed': 5
                # }
                # bbox_id = bbox["id"]
                # scaled_bbox = self.scale_bbox_to_detection(
                #         (bbox['x'], bbox['y'], bbox['w'], bbox['h'], bbox['time'], bbox['speed']),
                #         scale_x=6.3375,
                #         scale_y=6.3333
                #     )
                # self.imx500.set_inference_roi_abs(scaled_bbox)
                # self.current_bbox_id = bbox_id

                if not bbox_queue.empty():
                    bbox = bbox_queue.get()
                    bbox_id = bbox["id"]
                    if bbox_id not in self.processed_bbox_ids:
                        scaled_bbox = self.scale_bbox_to_detection(
                            (bbox['x'], bbox['y'], bbox['w'], bbox['h'], bbox['time'], bbox['speed']),
                            scale_x=6.3375,
                            scale_y=6.3333
                        )
                        self.imx500.set_inference_roi_abs(scaled_bbox)
                        self.current_bbox_id = bbox_id
                        self.frame_state = 1
                        self.detection_time = bbox['time']
                    else:
                        print(f"Skipping already processed bbox ID: {bbox_id}")
            elif self.frame_state == 1: # wait
                self.draw_roi(request)
                self.frame_state = 2
            else: # anormaly detection
                if self.current_bbox_id is not None:
                    results = self.process_anomaly_results(request)
                    if results:
                        result_dict = {
                            "id": results.bbox_id,
                            "is_anomaly": results.image_score > self.image_threshold,
                            "anomaly_score": results.image_score,
                        }
                        results_queue.put(result_dict)
                        print(f"Added to results_queue: {result_dict}")
                        self.draw_anomaly_results(request, results)
                        self.processed_bbox_ids.add(self.current_bbox_id)
                        self.current_bbox_id = None
                self.frame_state = 0
        except Exception as e:
            print(f"Error in process_frame: {e}")

def anomaly_process(bbox_queue, results_queue, args):
    detector = IMX500AnomalyDetector(args)
    detector.picam2.pre_callback = lambda req: detector.process_frame(
        req, bbox_queue, results_queue)
    while True:
        time.sleep(0.1)