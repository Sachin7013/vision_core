import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


class YOLOv8PoseDetector:
    """
    Detects human poses in video frames using YOLOv8 pose model.
    Draws skeleton keypoints and saves frames when poses are detected.
    """
    def __init__(self, conf: float = 0.5, device: str = "cpu"):
        self.device = device
        print(f"[detector] Loading YOLOv8 Pose Detection model...")
        
        # Load the pre-trained YOLOv8 medium pose model
        self.model = YOLO("yolov8m-pose.pt")
        self.model.to(self.device)
        
        self.conf = conf  # Confidence threshold for detections
        self.pose_detected = False  # Flag to track if pose was detected in current frame
        self.detection_count = 0  # Counter for total detections
        
        # Create directory to save detected frames
        self.frames_dir = Path("detected_frames")
        self.frames_dir.mkdir(exist_ok=True)
        print(f"[detector] ✅ YOLOv8 Pose model loaded successfully")
        print(f"[detector] Detected frames will be saved to: {self.frames_dir.absolute()}")
        
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        self.skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]
        
        self.skeleton_color = (0, 255, 255)
        self.keypoint_color = (0, 255, 0)
        self.keypoint_radius = 5

    def annotate(self, bgr: np.ndarray) -> tuple:
        """
        Run pose detection on a frame and draw skeleton keypoints.
        
        Args:
            bgr: Input frame in BGR format (OpenCV format)
            
        Returns:
            tuple: (annotated_frame, pose_detected_flag)
        """
        # Validate input frame
        if bgr is None or bgr.size == 0:
            return bgr, False
            
        try:
            h, w = bgr.shape[:2]
            output = bgr.copy()
            
            # Run YOLOv8 pose detection on the frame
            results = self.model(bgr, conf=self.conf, verbose=False)
            
            pose_detected = False
            
            # Check if any poses were detected
            if results and len(results) > 0:
                result = results[0]
                
                # If keypoints found, draw skeleton on frame
                if result.keypoints is not None and len(result.keypoints) > 0:
                    self.detection_count += 1
                    print(f"[detector] 👤 POSE DETECTED! Count: {self.detection_count}")
                    pose_detected = True
                    
                    # Extract keypoint coordinates and confidence scores
                    keypoints = result.keypoints.xy.cpu().numpy()
                    confidences = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
                    
                    # Draw skeleton for each detected person
                    for person_idx, person_keypoints in enumerate(keypoints):
                        person_conf = confidences[person_idx] if confidences is not None else None
                        
                        # Draw skeleton lines connecting keypoints
                        for skeleton_pair in self.skeleton:
                            pt1_idx, pt2_idx = skeleton_pair[0] - 1, skeleton_pair[1] - 1
                            
                            if pt1_idx < len(person_keypoints) and pt2_idx < len(person_keypoints):
                                pt1 = person_keypoints[pt1_idx]
                                pt2 = person_keypoints[pt2_idx]
                                
                                # Only draw if both points are valid (positive coordinates)
                                if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                                    pt1 = (int(pt1[0]), int(pt1[1]))
                                    pt2 = (int(pt2[0]), int(pt2[1]))
                                    cv2.line(output, pt1, pt2, self.skeleton_color, 2)
                        
                        # Draw keypoint circles
                        for kpt_idx, keypoint in enumerate(person_keypoints):
                            if keypoint[0] > 0 and keypoint[1] > 0:
                                kpt_conf = person_conf[kpt_idx] if person_conf is not None else 1.0
                                # Only draw keypoint if confidence is above threshold
                                if kpt_conf > 0.3:
                                    pt = (int(keypoint[0]), int(keypoint[1]))
                                    # Draw filled circle (green) with white border
                                    cv2.circle(output, pt, self.keypoint_radius, self.keypoint_color, -1)
                                    cv2.circle(output, pt, self.keypoint_radius, (255, 255, 255), 1)
                    
                    # Add text label to frame
                    cv2.putText(output, "POSE DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
                    
                    # Save the annotated frame to disk
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        filename = self.frames_dir / f"pose_{self.detection_count}_{timestamp}.jpg"
                        cv2.imwrite(str(filename), output)
                        print(f"[detector] 💾 Frame saved: {filename}")
                    except Exception as e:
                        print(f"[detector] ⚠️ Failed to save frame: {e}")
            
            # Update pose detection flag
            self.pose_detected = pose_detected
            return output, pose_detected
            
        except Exception as e:
            print(f"[detector] ❌ Error in annotate: {e}")
            import traceback
            traceback.print_exc()
            return bgr, False


def load_detector_from_env():
    enable_detection = os.getenv("ENABLE_DETECTION", "0")
    if enable_detection != "1":
        return None
    conf = float(os.getenv("DETECTION_CONF", "0.5"))
    device = os.getenv("DETECTION_DEVICE", "cpu")
    return YOLOv8PoseDetector(conf=conf, device=device)

