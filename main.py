"""
Parking Occupancy Detection: Traditional CV vs YOLO Comparison
================================================================

This script compares two methods for detecting parking occupancy from bird's-eye view video:
1. Traditional CV: Optical flow + affine stabilization + edge density + intensity analysis
2. YOLO-based: YOLOv8 object detection + IoU-based slot matching

Usage:
    python compare_methods.py --video parking_1920_1080.mp4 --slots parking_slots_2.json --out results/
    
    Optional flags:
        --frames N          Process only first N frames (default: all)
        --no-viz            Skip video visualization windows
        --gt-frames N       Number of frames to manually label for ground truth (default: 0)

Requirements:
    pip install opencv-python ultralytics numpy matplotlib seaborn scikit-learn torch torchvision

Output:
    - output_traditional.mp4: Traditional method visualization
    - output_yolo.mp4: YOLO method visualization
    - output_comparison.mp4: Side-by-side comparison
    - metrics.json: Quantitative results
    - plots/*.png: Various comparison charts
"""

import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import sys

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    print("ERROR: Please install ultralytics and torch:")
    print("  pip install ultralytics torch torchvision")
    sys.exit(1)

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


class ParkingSlot:
    """Represents a single parking slot with tracking state."""
    
    def __init__(self, slot_id: str, points: np.ndarray):
        self.device = 'cpu'
        self.id = slot_id
        self.points_ref = np.array(points, dtype=np.float32)  # Reference polygon
        
        # Calculate center for labeling
        M_moment = cv2.moments(np.int32(self.points_ref))
        self.cx = int(M_moment['m10'] / M_moment['m00']) if M_moment['m00'] != 0 else 0
        self.cy = int(M_moment['m01'] / M_moment['m00']) if M_moment['m00'] != 0 else 0
        self.center_ref = np.array([[[self.cx, self.cy]]], dtype=np.float32)
        
        # State tracking for temporal filtering
        self.frames_occupied_trad = 0
        self.frames_empty_trad = 0
        self.status_trad = False
        
        self.frames_occupied_yolo = 0
        self.frames_empty_yolo = 0
        self.status_yolo = False
        
        # Ground truth (if available)
        self.gt_status = None
    
    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Get axis-aligned bounding box of the slot polygon."""
        x, y, w, h = cv2.boundingRect(np.int32(self.points_ref))
        return x, y, w, h


class TraditionalDetector:
    """
    Traditional CV method using optical flow stabilization + edge/intensity analysis.
    
    Assumptions:
    - Camera is mostly static (minor jitter handled by affine stabilization)
    - Parking slots are fixed in the scene
    - Occupied slots have higher edge density (car outlines) and intensity variance
    
    Limitations:
    - Sensitive to lighting changes and shadows
    - Requires parameter tuning for different scenes
    - May fail with unusual parking patterns or occlusions
    """
    
    def __init__(self, slots: List[ParkingSlot], width: int, height: int):
        self.slots = slots
        self.width = width
        self.height = height
        
        # Optical flow parameters
        self.feature_params = dict(
            maxCorners=1000,
            qualityLevel=0.03,
            minDistance=30,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Stabilization state
        self.smoothed_M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        self.SMOOTHING_ALPHA = 0.15
        
        self.p0 = None  # Current tracked points
        self.p_initial = None  # Initial positions of tracked points
        self.old_gray = None
        
        # Detection thresholds
        self.EDGE_THRESHOLD = 0.15  # Minimum edge density for occupied
        self.STD_THRESHOLD = 35.0   # Minimum std dev for occupied
        self.REQUIRED_FRAMES = 10   # Consecutive frames for status change
        
    def initialize(self, first_frame: np.ndarray):
        """Initialize optical flow tracking on first frame."""
        self.old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
        self.p_initial = self.p0.copy()
        
    def update_stabilization(self, frame: np.ndarray) -> np.ndarray:
        """
        Update affine transformation to stabilize frame using optical flow.
        
        Process:
        1. Track features using Lucas-Kanade optical flow
        2. Estimate affine transform using RANSAC
        3. Apply exponential smoothing to reduce jitter
        4. Refresh features when count drops
        
        Returns:
            gray: Grayscale version of current frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.p0 is not None and len(self.p0) > 10:
            # Track features
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                self.old_gray, gray, self.p0, None, **self.lk_params
            )
            
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            good_initial = self.p_initial[st == 1]
            
            if len(good_new) >= 4:
                # Estimate affine transformation
                M, inliers = cv2.estimateAffinePartial2D(
                    good_initial, good_new, cv2.RANSAC
                )
                
                if M is not None:
                    # Smooth transformation to reduce jitter
                    self.smoothed_M = (
                        self.SMOOTHING_ALPHA * M + 
                        (1.0 - self.SMOOTHING_ALPHA) * self.smoothed_M
                    )
            
            # Update tracking state
            self.old_gray = gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            self.p_initial = good_initial.reshape(-1, 1, 2)
            
            # Refresh features if running low
            if len(self.p0) < 200:
                new_features = cv2.goodFeaturesToTrack(
                    gray, mask=None, **self.feature_params
                )
                if new_features is not None:
                    M_inv = cv2.invertAffineTransform(self.smoothed_M)
                    new_initials = cv2.transform(new_features, M_inv)
                    self.p0 = np.vstack((self.p0, new_features))
                    self.p_initial = np.vstack((self.p_initial, new_initials))
        
        return gray
    
    def detect(self, frame: np.ndarray) -> Dict[str, bool]:
        """
        Detect occupancy for all slots using edge density and intensity variance.
        
        For each slot:
        1. Transform reference polygon using current affine matrix
        2. Extract ROI and create mask
        3. Compute edge density (Canny edges)
        4. Compute intensity standard deviation
        5. Apply thresholds and temporal filtering
        
        Returns:
            Dictionary mapping slot_id to occupancy status (True=occupied)
        """
        # Update stabilization
        gray = self.update_stabilization(frame)
        
        # Compute edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 120, 220)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, dilate_kernel, iterations=1)
        
        results = {}
        
        for slot in self.slots:
            # Transform slot polygon to current frame
            pts_ref_reshaped = np.array([slot.points_ref])
            pts_curr = cv2.transform(pts_ref_reshaped, self.smoothed_M)[0]
            pts_curr = np.int32(pts_curr)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(pts_curr)
            x, y = max(0, x), max(0, y)
            w = min(self.width - x, w)
            h = min(self.height - y, h)
            
            if w <= 0 or h <= 0:
                results[slot.id] = slot.status_trad
                continue
            
            # Extract ROI
            roi_edges = edges[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]
            
            # Create mask for polygon
            local_mask = np.zeros((h, w), dtype=np.uint8)
            local_pts = pts_curr - [x, y]
            cv2.fillPoly(local_mask, [local_pts], 255)
            
            valid_pixels = (local_mask == 255)
            total_pixels = np.count_nonzero(valid_pixels)
            
            if total_pixels == 0:
                results[slot.id] = slot.status_trad
                continue
            
            # Compute metrics
            edge_density = np.count_nonzero(roi_edges[valid_pixels]) / total_pixels
            std_dev = np.std(roi_gray[valid_pixels])
            
            # Determine occupancy
            is_occupied_now = (
                edge_density > self.EDGE_THRESHOLD or 
                std_dev > self.STD_THRESHOLD
            )
            
            # Temporal filtering
            if is_occupied_now:
                slot.frames_occupied_trad += 1
                slot.frames_empty_trad = 0
                if slot.frames_occupied_trad >= self.REQUIRED_FRAMES:
                    slot.status_trad = True
            else:
                slot.frames_empty_trad += 1
                slot.frames_occupied_trad = 0
                if slot.frames_empty_trad >= self.REQUIRED_FRAMES:
                    slot.status_trad = False
            
            results[slot.id] = slot.status_trad
        
        return results


# class YOLODetector:
#     """
#     YOLO-based method using pre-trained YOLOv8 for car detection.
    
#     Assumptions:
#     - Pre-trained COCO model can detect vehicles accurately
#     - IoU overlap indicates slot occupancy
#     - Camera view is stable (bird's eye view)
    
#     Limitations:
#     - May miss partially occluded vehicles
#     - Slower than traditional CV (especially on CPU)
#     - Requires GPU for real-time performance
#     - May detect vehicles outside parking area
#     """
    
#     def __init__(self, slots: List[ParkingSlot], model_name: str = 'yolov8n.pt'):
#         self.slots = slots
        
#         # Initialize YOLO
#         print(f"Loading YOLO model: {model_name}")
#         self.model = YOLO(model_name)
        
#         # Set device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
#         if torch.backends.mps.is_available():
#             self.device = 'mps'
#             print("Using Apple Silicon GPU (MPS)")
#         elif torch.cuda.is_available():
#             self.device = 'cuda'
#             print("Using NVIDIA GPU (CUDA)")
#         else:
#             self.device = 'cpu'
#             print("Using CPU (consider GPU for better performance)")
        
#         # COCO class IDs for vehicles
#         self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        
#         # Detection parameters
#         self.IOU_THRESHOLD = 0.3  # Minimum IoU to consider slot occupied
#         self.CONFIDENCE_THRESHOLD = 0.25  # Minimum detection confidence
#         self.REQUIRED_FRAMES = 10  # Consecutive frames for status change
        
#     def compute_polygon_iou(self, poly: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
#         """
#         Compute IoU between a polygon (slot) and a bounding box (detection).
        
#         Uses rasterization approach:
#         1. Create binary masks for both shapes
#         2. Compute intersection and union
#         3. Return IoU = intersection / union
#         """
#         x1, y1, x2, y2 = bbox
        
#         # Get bounding box of polygon
#         poly_bbox = cv2.boundingRect(np.int32(poly))
#         px, py, pw, ph = poly_bbox
        
#         # Determine combined bounding box
#         min_x = min(x1, px)
#         min_y = min(y1, py)
#         max_x = max(x2, px + pw)
#         max_y = max(y2, py + ph)
        
#         width = max_x - min_x
#         height = max_y - min_y
        
#         if width <= 0 or height <= 0:
#             return 0.0
        
#         # Create masks
#         mask_poly = np.zeros((height, width), dtype=np.uint8)
#         mask_bbox = np.zeros((height, width), dtype=np.uint8)
        
#         # Fill polygon mask
#         local_poly = np.int32(poly) - [min_x, min_y]
#         cv2.fillPoly(mask_poly, [local_poly], 255)
        
#         # Fill bbox mask
#         local_bbox = (x1 - min_x, y1 - min_y, x2 - min_x, y2 - min_y)
#         cv2.rectangle(
#             mask_bbox,
#             (local_bbox[0], local_bbox[1]),
#             (local_bbox[2], local_bbox[3]),
#             255,
#             -1
#         )
        
#         # Compute IoU
#         intersection = np.count_nonzero(np.logical_and(mask_poly, mask_bbox))
#         # union = np.count_nonzero(np.logical_or(mask_poly, mask_bbox))
#         # Tính diện tích gốc của slot thay vì Union
#         area_poly = np.count_nonzero(mask_poly)
#         return intersection / area_poly if area_poly > 0 else 0.0
        
#         # return intersection / union if union > 0 else 0.0
    
#     def detect(self, frame: np.ndarray) -> Dict[str, bool]:
#         """
#         Detect occupancy using YOLO car detection and IoU matching.
        
#         Process:
#         1. Run YOLO inference on frame
#         2. Filter detections by class (vehicles) and confidence
#         3. For each slot, compute IoU with all detections
#         4. Slot is occupied if max IoU > threshold
#         5. Apply temporal filtering
        
#         Returns:
#             Dictionary mapping slot_id to occupancy status
#         """
#         # Run YOLO inference
#         results = self.model(
#             frame,
#             device=self.device,
#             verbose=False,
#             imgsz=1280,
#             conf=self.CONFIDENCE_THRESHOLD
#         )[0]

#         cv2.imshow("YOLO Raw Detections", results.plot())
#         cv2.waitKey(1)
#         # -----------------
        
#         # Extract vehicle detections
#         detections = []
#         if results.boxes is not None:
#             for box in results.boxes:
#                 cls = int(box.cls[0])
#                 if cls in self.vehicle_classes:
#                     x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
#                     detections.append((int(x1), int(y1), int(x2), int(y2)))
        
#         # Match detections to slots
#         slot_results = {}
        
#         for slot in self.slots:
#             max_iou = 0.0
            
#             # Compute IoU with all detections
#             for det_bbox in detections:
#                 iou = self.compute_polygon_iou(slot.points_ref, det_bbox)
#                 max_iou = max(max_iou, iou)
            
#             # Determine occupancy
#             is_occupied_now = max_iou > self.IOU_THRESHOLD
            
#             # Temporal filtering
#             # if is_occupied_now:
#             #     slot.frames_occupied_yolo += 1
#             #     slot.frames_empty_yolo = 0
#             #     if slot.frames_occupied_yolo >= self.REQUIRED_FRAMES:
#             #         slot.status_yolo = True
#             # else:
#             #     slot.frames_empty_yolo += 1
#             #     slot.frames_occupied_yolo = 0
#             #     if slot.frames_empty_yolo >= self.REQUIRED_FRAMES:
#             #         slot.status_yolo = False

#             if is_occupied_now:
#                 slot.frames_occupied_yolo = min(slot.frames_occupied_yolo + 1, self.REQUIRED_FRAMES)
#                 slot.frames_empty_yolo = 0
#                 if slot.frames_occupied_yolo >= self.REQUIRED_FRAMES:
#                     slot.status_yolo = True
#             else:
#                 slot.frames_empty_yolo += 1
#                 if slot.frames_empty_yolo >= self.REQUIRED_FRAMES:
#                     slot.status_yolo = False
#                     slot.frames_occupied_yolo = 0 # Chỉ reset khi chắc chắn xe đã rời đi
            
#             slot_results[slot.id] = slot.status_yolo
        
#         return slot_results

import os
try:
    from inference import get_model
except ImportError:
    print("ERROR: Please install Roboflow inference:")
    print("  pip install inference")
    sys.exit(1)

class YOLODetector:
    """
    YOLO-based method using Roboflow Inference API.
    """
    
    def __init__(self, slots: List[ParkingSlot], model_name: str = 'parking-lot-j4ojc-9jaft/1'):
        self.slots = slots
        
        print(f"Loading Roboflow model: {model_name}")
        
        # BƯỚC QUAN TRỌNG: Bạn cần paste API Key của bạn vào đây.
        # Lấy ở web Roboflow (chỗ bạn vừa copy code có dòng export ROBOFLOW_API_KEY=...)
        # os.environ["ROBOFLOW_API_KEY"] = "R9m7j4NkesxiRS1BXJYi"
        my_api_key = "R9m7j4NkesxiRS1BXJYi"
        
        # Explicitly pass the api_key parameter to get_model
        self.model = get_model(model_id=model_name, api_key=my_api_key)
        
        # Tải mô hình từ Roboflow
        # self.model = get_model(model_id=model_name)
        
        # Detection parameters
        self.IOU_THRESHOLD = 0.3  # Tỷ lệ diện tích giao nhau tối thiểu
        self.CONFIDENCE_THRESHOLD = 0.25  # Độ tin cậy tối thiểu
        self.REQUIRED_FRAMES = 10  # Số frame liên tiếp để chuyển trạng thái
        
    def compute_polygon_iou(self, poly: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """
        Compute IoU between a polygon (slot) and a bounding box (detection).
        (GIỮ NGUYÊN LOGIC RASTERIZATION CỦA BẠN)
        """
        x1, y1, x2, y2 = bbox
        
        poly_bbox = cv2.boundingRect(np.int32(poly))
        px, py, pw, ph = poly_bbox
        
        min_x = min(x1, px)
        min_y = min(y1, py)
        max_x = max(x2, px + pw)
        max_y = max(y2, py + ph)
        
        width = max_x - min_x
        height = max_y - min_y
        
        if width <= 0 or height <= 0:
            return 0.0
        
        mask_poly = np.zeros((height, width), dtype=np.uint8)
        mask_bbox = np.zeros((height, width), dtype=np.uint8)
        
        local_poly = np.int32(poly) - [min_x, min_y]
        cv2.fillPoly(mask_poly, [local_poly], 255)
        
        local_bbox = (x1 - min_x, y1 - min_y, x2 - min_x, y2 - min_y)
        cv2.rectangle(
            mask_bbox,
            (local_bbox[0], local_bbox[1]),
            (local_bbox[2], local_bbox[3]),
            255,
            -1
        )
        
        intersection = np.count_nonzero(np.logical_and(mask_poly, mask_bbox))
        area_poly = np.count_nonzero(mask_poly)
        return intersection / area_poly if area_poly > 0 else 0.0
    
    def detect(self, frame: np.ndarray) -> Dict[str, bool]:
        """
        Detect occupancy using Roboflow inference and IoU matching.
        """
        # 1. Run inference qua thư viện Roboflow
        results = self.model.infer(frame)[0]
        
        # 2. Extract detections và chuyển đổi format bounding box
        detections = []
        for pred in results.predictions:
            if pred.confidence >= self.CONFIDENCE_THRESHOLD:
                # Roboflow trả về tâm x, y cùng với width, height
                w = pred.width
                h = pred.height
                x_center = pred.x
                y_center = pred.y
                
                # Chuyển đổi về x1, y1 (góc trái trên) và x2, y2 (góc phải dưới)
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)
                
                detections.append((x1, y1, x2, y2))
        
        # 3. Match detections to slots
        slot_results = {}
        
        for slot in self.slots:
            max_iou = 0.0
            
            for det_bbox in detections:
                iou = self.compute_polygon_iou(slot.points_ref, det_bbox)
                max_iou = max(max_iou, iou)
            
            is_occupied_now = max_iou > self.IOU_THRESHOLD
            
            # Temporal filtering (Lọc nhiễu theo thời gian)
            if is_occupied_now:
                slot.frames_occupied_yolo = min(slot.frames_occupied_yolo + 1, self.REQUIRED_FRAMES)
                slot.frames_empty_yolo = 0
                if slot.frames_occupied_yolo >= self.REQUIRED_FRAMES:
                    slot.status_yolo = True
            else:
                slot.frames_empty_yolo += 1
                if slot.frames_empty_yolo >= self.REQUIRED_FRAMES:
                    slot.status_yolo = False
                    slot.frames_occupied_yolo = 0 
            
            slot_results[slot.id] = slot.status_yolo
        
        return slot_results


def visualize_frame(
    frame: np.ndarray,
    slots: List[ParkingSlot],
    results: Dict[str, bool],
    method_name: str
) -> np.ndarray:
    """Draw slot polygons and labels on frame."""
    vis_frame = frame.copy()
    
    occupied_count = 0
    empty_count = 0
    
    for slot in slots:
        status = results[slot.id]
        pts = np.int32(slot.points_ref)
        
        if status:
            color = (0, 0, 255)  # Red for occupied
            occupied_count += 1
            # Draw slot number
            slot_num = slot.id.split('_')[1]
            cv2.putText(
                vis_frame,
                slot_num,
                (slot.cx - 10, slot.cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
        else:
            color = (0, 255, 0)  # Green for empty
            empty_count += 1
        
        cv2.polylines(vis_frame, [pts], isClosed=True, color=color, thickness=2)
    
    # Add info text
    cv2.putText(
        vis_frame,
        f"{method_name}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )
    cv2.putText(
        vis_frame,
        f"Occupied: {occupied_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )
    cv2.putText(
        vis_frame,
        f"Empty: {empty_count}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    return vis_frame


def create_comparison_frame(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """Create side-by-side comparison."""
    h, w = frame1.shape[:2]
    comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
    comparison[:, :w] = frame1
    comparison[:, w:] = frame2
    
    # Add divider
    cv2.line(comparison, (w, 0), (w, h), (255, 255, 255), 3)
    
    return comparison


def save_plots(
    metrics: Dict,
    output_dir: Path,
    has_ground_truth: bool
):
    """Generate and save comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Occupancy over time
    plt.figure(figsize=(12, 6))
    frames = range(len(metrics['occupancy_trad']))
    plt.plot(frames, metrics['occupancy_trad'], label='Traditional', alpha=0.7)
    plt.plot(frames, metrics['occupancy_yolo'], label='YOLO', alpha=0.7)
    plt.xlabel('Frame')
    plt.ylabel('Occupied Slots')
    plt.title('Parking Occupancy Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'occupancy_timeline.png', dpi=150)
    plt.close()
    
    # 2. FPS comparison
    plt.figure(figsize=(8, 6))
    methods = ['Traditional', 'YOLO']
    fps_values = [metrics['fps_trad'], metrics['fps_yolo']]
    colors = ['#3498db', '#e74c3c']
    plt.bar(methods, fps_values, color=colors)
    plt.ylabel('Frames Per Second')
    plt.title('Processing Speed Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(fps_values):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'fps_comparison.png', dpi=150)
    plt.close()
    
    # 3. Agreement heatmap (confusion between methods)
    if 'agreement_cm' in metrics:
        plt.figure(figsize=(8, 6))
        cm = metrics['agreement_cm']
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Empty (YOLO)', 'Occupied (YOLO)'],
            yticklabels=['Empty (Trad)', 'Occupied (Trad)']
        )
        plt.title('Method Agreement Matrix\n(Traditional vs YOLO as reference)')
        plt.ylabel('Traditional Method')
        plt.xlabel('YOLO Method')
        plt.tight_layout()
        plt.savefig(output_dir / 'method_agreement.png', dpi=150)
        plt.close()
    
    # 4. Ground truth metrics (if available)
    if has_ground_truth and 'gt_metrics_trad' in metrics:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Traditional method metrics
        met_trad = metrics['gt_metrics_trad']
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1']
        metrics_values_trad = [
            met_trad['accuracy'],
            met_trad['precision'],
            met_trad['recall'],
            met_trad['f1']
        ]
        
        ax1.bar(metrics_names, metrics_values_trad, color='#3498db')
        ax1.set_ylim([0, 1])
        ax1.set_title('Traditional Method Performance')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(metrics_values_trad):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # YOLO method metrics
        met_yolo = metrics['gt_metrics_yolo']
        metrics_values_yolo = [
            met_yolo['accuracy'],
            met_yolo['precision'],
            met_yolo['recall'],
            met_yolo['f1']
        ]
        
        ax2.bar(metrics_names, metrics_values_yolo, color='#e74c3c')
        ax2.set_ylim([0, 1])
        ax2.set_title('YOLO Method Performance')
        ax2.set_ylabel('Score')
        ax2.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(metrics_values_yolo):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gt_performance.png', dpi=150)
        plt.close()
        
        # Confusion matrices for ground truth
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.heatmap(
            met_trad['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Empty', 'Occupied'],
            yticklabels=['Empty', 'Occupied'],
            ax=ax1
        )
        ax1.set_title('Traditional Method vs Ground Truth')
        ax1.set_ylabel('Predicted')
        ax1.set_xlabel('Ground Truth')
        
        sns.heatmap(
            met_yolo['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Reds',
            xticklabels=['Empty', 'Occupied'],
            yticklabels=['Empty', 'Occupied'],
            ax=ax2
        )
        ax2.set_title('YOLO Method vs Ground Truth')
        ax2.set_ylabel('Predicted')
        ax2.set_xlabel('Ground Truth')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'gt_confusion_matrices.png', dpi=150)
        plt.close()
    
    print(f"✓ Plots saved to {output_dir}")


def interactive_ground_truth_labeling(
    video_path: str,
    slots: List[ParkingSlot],
    num_frames: int
) -> Dict[int, Dict[str, bool]]:
    """
    Interactive labeling of ground truth for evaluation.
    
    User presses keys to label each slot:
    - 'o': occupied
    - 'e': empty
    - 's': skip this frame
    - 'q': quit labeling
    
    Returns:
        Dictionary mapping frame_number -> {slot_id: bool}
    """
    print("\n" + "="*70)
    print("GROUND TRUTH LABELING")
    print("="*70)
    print(f"You will label {num_frames} random frames.")
    print("\nFor each frame:")
    print("  - Each slot will be highlighted in YELLOW")
    print("  - Press 'o' if OCCUPIED, 'e' if EMPTY")
    print("  - Press 's' to skip current frame")
    print("  - Press 'q' to quit labeling")
    print("="*70)
    input("Press ENTER to start...")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Select random frames
    np.random.seed(42)
    frame_indices = sorted(np.random.choice(
        range(50, total_frames - 50),
        size=min(num_frames, total_frames - 100),
        replace=False
    ))
    
    ground_truth = {}
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_labels = {}
        skip_frame = False
        
        for slot in slots:
            while True:
                vis_frame = frame.copy()
                
                # Draw all slots in gray
                for s in slots:
                    pts = np.int32(s.points_ref)
                    color = (128, 128, 128) if s.id != slot.id else (0, 255, 255)
                    thickness = 2 if s.id != slot.id else 4
                    cv2.polylines(vis_frame, [pts], True, color, thickness)
                
                # Highlight current slot
                pts = np.int32(slot.points_ref)
                cv2.polylines(vis_frame, [pts], True, (0, 255, 255), 4)
                cv2.putText(
                    vis_frame,
                    f"Frame {frame_idx} - Slot {slot.id}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    3
                )
                cv2.putText(
                    vis_frame,
                    "o=Occupied | e=Empty | s=Skip frame | q=Quit",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )
                
                cv2.imshow('Ground Truth Labeling', cv2.resize(vis_frame, (1280, 720)))
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('o'):
                    frame_labels[slot.id] = True
                    break
                elif key == ord('e'):
                    frame_labels[slot.id] = False
                    break
                elif key == ord('s'):
                    skip_frame = True
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return ground_truth
            
            if skip_frame:
                break
        
        if not skip_frame:
            ground_truth[frame_idx] = frame_labels
            print(f"✓ Labeled frame {frame_idx} ({len(ground_truth)}/{num_frames})")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✓ Labeling complete! {len(ground_truth)} frames labeled.")
    return ground_truth


def main():
    parser = argparse.ArgumentParser(
        description='Compare Traditional CV vs YOLO for parking occupancy detection'
    )
    parser.add_argument(
        '--video',
        default='parking_1920_1080.mp4',
        help='Path to video file'
    )
    parser.add_argument(
        '--slots',
        default='parking_slots.json',
        help='Path to parking slots JSON file'
    )
    parser.add_argument(
        '--out',
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--frames',
        type=int,
        default=None,
        help='Process only first N frames (default: all)'
    )
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Skip visualization windows'
    )
    parser.add_argument(
        '--gt-frames',
        type=int,
        default=0,
        help='Number of frames to manually label for ground truth evaluation'
    )
    parser.add_argument(
        '--yolo-model',
        default='yolov8n.pt',
        help='YOLO model to use (yolov8n.pt, yolov8s.pt, etc.)'
    )
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("PARKING OCCUPANCY DETECTION COMPARISON")
    print("="*70)
    
    # Load parking slots
    print(f"\n[1/7] Loading parking slots from {args.slots}...")
    try:
        with open(args.slots, 'r', encoding='utf-8') as f:
            slots_data = json.load(f)['parking_slots']
        slots = [ParkingSlot(s['id'], s['points']) for s in slots_data]
        print(f"✓ Loaded {len(slots)} parking slots")
    except Exception as e:
        print(f"✗ Error loading slots: {e}")
        return
    
    # Open video
    print(f"\n[2/7] Opening video {args.video}...")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"✗ Error: Cannot open video file")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✓ Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
    
    # Determine frames to process
    max_frames = args.frames if args.frames else total_frames
    max_frames = min(max_frames, total_frames)
    
    # Ground truth labeling (if requested)
    ground_truth = {}
    if args.gt_frames > 0:
        print(f"\n[3/7] Ground truth labeling ({args.gt_frames} frames)...")
        ground_truth = interactive_ground_truth_labeling(
            args.video,
            slots,
            args.gt_frames
        )
    else:
        print(f"\n[3/7] Skipping ground truth labeling (use --gt-frames N to enable)")
    
    # Initialize detectors
    print(f"\n[4/7] Initializing detectors...")
    trad_detector = TraditionalDetector(slots, width, height)
    yolo_detector = YOLODetector(slots, model_name=args.yolo_model)
    
    # Initialize first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, first_frame = cap.read()
    if not ret:
        print("✗ Error reading first frame")
        return
    
    trad_detector.initialize(first_frame)
    print("✓ Detectors initialized")
    
    # Setup video writers
    print(f"\n[5/7] Setting up output videos...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_trad = cv2.VideoWriter(
        str(output_dir / 'output_traditional.mp4'),
        fourcc, fps, (width, height)
    )
    out_yolo = cv2.VideoWriter(
        str(output_dir / 'output_yolo.mp4'),
        fourcc, fps, (width, height)
    )
    out_comp = cv2.VideoWriter(
        str(output_dir / 'output_comparison.mp4'),
        fourcc, fps, (width * 2, height)
    )
    
    # Process video
    print(f"\n[6/7] Processing {max_frames} frames...")
    print("Progress: ", end='', flush=True)
    
    # Metrics collection
    metrics = {
        'occupancy_trad': [],
        'occupancy_yolo': [],
        'frame_times_trad': [],
        'frame_times_yolo': [],
        'all_results_trad': [],
        'all_results_yolo': [],
        'gt_frames': []
    }
    
    frame_count = 50
    progress_interval = max(1, max_frames // 50)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Traditional method
        t_start = time.time()
        results_trad = trad_detector.detect(frame)
        t_trad = time.time() - t_start
        
        # YOLO method
        t_start = time.time()
        results_yolo = yolo_detector.detect(frame)
        t_yolo = time.time() - t_start
        
        # Collect metrics
        occupied_trad = sum(results_trad.values())
        occupied_yolo = sum(results_yolo.values())
        
        metrics['occupancy_trad'].append(occupied_trad)
        metrics['occupancy_yolo'].append(occupied_yolo)
        metrics['frame_times_trad'].append(t_trad)
        metrics['frame_times_yolo'].append(t_yolo)
        
        # Store all results for later analysis
        metrics['all_results_trad'].append(results_trad.copy())
        metrics['all_results_yolo'].append(results_yolo.copy())
        
        # Check if this frame has ground truth
        if frame_count in ground_truth:
            metrics['gt_frames'].append({
                'frame_idx': frame_count,
                'gt': ground_truth[frame_count],
                'trad': results_trad.copy(),
                'yolo': results_yolo.copy()
            })
        
        # Visualize
        vis_trad = visualize_frame(frame, slots, results_trad, "Traditional CV")
        vis_yolo = visualize_frame(frame, slots, results_yolo, "YOLO Detection")
        vis_comp = create_comparison_frame(vis_trad, vis_yolo)
        
        # Write outputs
        out_trad.write(vis_trad)
        out_yolo.write(vis_yolo)
        out_comp.write(vis_comp)
        
        # Display (if enabled)
        if not args.no_viz:
            cv2.imshow('Comparison', cv2.resize(vis_comp, (1280, 360)))
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
        
        # Progress
        if i % progress_interval == 0:
            print('▓', end='', flush=True)
        
        frame_count += 1
    
    print(" ✓")
    
    # Cleanup
    cap.release()
    out_trad.release()
    out_yolo.release()
    out_comp.release()
    cv2.destroyAllWindows()
    
    # Compute final metrics
    print(f"\n[7/7] Computing metrics and generating report...")
    
    # FPS
    metrics['fps_trad'] = 1.0 / np.mean(metrics['frame_times_trad'])
    metrics['fps_yolo'] = 1.0 / np.mean(metrics['frame_times_yolo'])
    
    # Agreement between methods (use YOLO as pseudo ground truth)
    y_trad_all = []
    y_yolo_all = []
    for res_trad, res_yolo in zip(metrics['all_results_trad'], metrics['all_results_yolo']):
        for slot_id in res_trad.keys():
            y_trad_all.append(1 if res_trad[slot_id] else 0)
            y_yolo_all.append(1 if res_yolo[slot_id] else 0)
    
    metrics['agreement_cm'] = confusion_matrix(y_trad_all, y_yolo_all)
    metrics['agreement_score'] = accuracy_score(y_trad_all, y_yolo_all)
    
    # Ground truth metrics (if available)
    if len(metrics['gt_frames']) > 0:
        # Traditional vs GT
        y_true_trad = []
        y_pred_trad = []
        
        # YOLO vs GT
        y_true_yolo = []
        y_pred_yolo = []
        
        for gt_frame in metrics['gt_frames']:
            gt = gt_frame['gt']
            trad = gt_frame['trad']
            yolo = gt_frame['yolo']
            
            for slot_id, gt_status in gt.items():
                y_true_trad.append(1 if gt_status else 0)
                y_pred_trad.append(1 if trad[slot_id] else 0)
                
                y_true_yolo.append(1 if gt_status else 0)
                y_pred_yolo.append(1 if yolo[slot_id] else 0)
        
        # Traditional metrics
        acc_trad = accuracy_score(y_true_trad, y_pred_trad)
        prec_trad, rec_trad, f1_trad, _ = precision_recall_fscore_support(
            y_true_trad, y_pred_trad, average='binary', zero_division=0
        )
        cm_trad = confusion_matrix(y_true_trad, y_pred_trad)
        
        metrics['gt_metrics_trad'] = {
            'accuracy': acc_trad,
            'precision': prec_trad,
            'recall': rec_trad,
            'f1': f1_trad,
            'confusion_matrix': cm_trad
        }
        
        # YOLO metrics
        acc_yolo = accuracy_score(y_true_yolo, y_pred_yolo)
        prec_yolo, rec_yolo, f1_yolo, _ = precision_recall_fscore_support(
            y_true_yolo, y_pred_yolo, average='binary', zero_division=0
        )
        cm_yolo = confusion_matrix(y_true_yolo, y_pred_yolo)
        
        metrics['gt_metrics_yolo'] = {
            'accuracy': acc_yolo,
            'precision': prec_yolo,
            'recall': rec_yolo,
            'f1': f1_yolo,
            'confusion_matrix': cm_yolo
        }
    
    # Save metrics to JSON
    metrics_json = {
        'video': args.video,
        'frames_processed': len(metrics['occupancy_trad']),
        'fps_traditional': metrics['fps_trad'],
        'fps_yolo': metrics['fps_yolo'],
        'agreement_score': metrics['agreement_score'],
        'mean_occupancy_traditional': float(np.mean(metrics['occupancy_trad'])),
        'mean_occupancy_yolo': float(np.mean(metrics['occupancy_yolo'])),
    }
    
    if 'gt_metrics_trad' in metrics:
        metrics_json['ground_truth_traditional'] = {
            k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in metrics['gt_metrics_trad'].items()
        }
        metrics_json['ground_truth_yolo'] = {
            k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in metrics['gt_metrics_yolo'].items()
        }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    # Generate plots
    save_plots(
        metrics,
        output_dir / 'plots',
        has_ground_truth=len(metrics['gt_frames']) > 0
    )
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\nFrames processed: {len(metrics['occupancy_trad'])}")
    print(f"\n{'Method':<20} {'FPS':<10} {'Avg Occupancy'}")
    print("-" * 50)
    print(f"{'Traditional CV':<20} {metrics['fps_trad']:<10.2f} {np.mean(metrics['occupancy_trad']):.1f}")
    print(f"{'YOLO':<20} {metrics['fps_yolo']:<10.2f} {np.mean(metrics['occupancy_yolo']):.1f}")
    
    print(f"\nMethod Agreement: {metrics['agreement_score']:.1%}")
    print(f"  (Percentage of slot-frames where both methods agree)")
    
    if 'gt_metrics_trad' in metrics:
        print(f"\n{'Ground Truth Evaluation'} ({len(metrics['gt_frames'])} labeled frames)")
        print("-" * 50)
        print(f"{'Method':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1'}")
        print("-" * 70)
        
        met_trad = metrics['gt_metrics_trad']
        print(f"{'Traditional':<20} {met_trad['accuracy']:<12.3f} {met_trad['precision']:<12.3f} {met_trad['recall']:<12.3f} {met_trad['f1']:.3f}")
        
        met_yolo = metrics['gt_metrics_yolo']
        print(f"{'YOLO':<20} {met_yolo['accuracy']:<12.3f} {met_yolo['precision']:<12.3f} {met_yolo['recall']:<12.3f} {met_yolo['f1']:.3f}")
    
    print(f"\n{'Outputs saved to:'} {output_dir}")
    print(f"  - output_traditional.mp4")
    print(f"  - output_yolo.mp4")
    print(f"  - output_comparison.mp4")
    print(f"  - metrics.json")
    print(f"  - plots/")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()