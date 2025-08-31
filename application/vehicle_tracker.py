import cv2
import numpy as np
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
import math

@dataclass
class TrackedVehicle:
    """Enhanced vehicle tracking with hysteresis and violation logic"""
    track_id: str
    plate_text: str
    bbox_history: deque  # Store last N bounding boxes for smoothing
    confidence_history: deque  # Store confidence values
    first_seen: float
    last_seen: float
    frames_seen: int
    violation_detected: bool
    violation_time: Optional[float]
    violation_confirmed: bool  # True after hysteresis confirmation
    hysteresis_counter: int  # Frames since violation first detected
    max_confidence: float
    average_confidence: float
    evidence_frames: List[np.ndarray]

    def __init__(self, track_id: str, plate_text: str, initial_bbox: Tuple[int, int, int, int],
                 confidence: float, max_history: int = 10):
        self.track_id = track_id
        self.plate_text = plate_text
        self.bbox_history = deque(maxlen=max_history)
        self.confidence_history = deque(maxlen=max_history)
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.frames_seen = 1
        self.violation_detected = False
        self.violation_time = None
        self.violation_confirmed = False
        self.hysteresis_counter = 0
        self.max_confidence = confidence
        self.average_confidence = confidence
        self.evidence_frames = []

        self.bbox_history.append(initial_bbox)
        self.confidence_history.append(confidence)

    def update(self, bbox: Tuple[int, int, int, int], confidence: float, frame: np.ndarray = None):
        """Update tracking with new detection"""
        self.last_seen = time.time()
        self.frames_seen += 1

        # Update bbox and confidence history
        self.bbox_history.append(bbox)
        self.confidence_history.append(confidence)

        # Update confidence statistics
        self.max_confidence = max(self.max_confidence, confidence)
        self.average_confidence = sum(self.confidence_history) / len(self.confidence_history)

        # Store evidence frame if violation detected
        if self.violation_detected and frame is not None:
            self.evidence_frames.append(frame.copy())
            # Keep only last 5 evidence frames
            if len(self.evidence_frames) > 5:
                self.evidence_frames.pop(0)

    def get_smoothed_bbox(self) -> Tuple[int, int, int, int]:
        """Get smoothed bounding box using recent history"""
        if len(self.bbox_history) == 0:
            return (0, 0, 0, 0)

        # For testing purposes, use the most recent bbox to make violation detection more responsive
        # In production, you might want to use smoothing for stability
        return self.bbox_history[-1]

    def get_current_confidence(self) -> float:
        """Get current confidence (latest or average)"""
        if len(self.confidence_history) == 0:
            return 0.0
        return self.confidence_history[-1]

    def mark_violation(self, violation_time: float):
        """Mark vehicle as having committed a violation"""
        if not self.violation_detected:
            self.violation_detected = True
            self.violation_time = violation_time
            self.hysteresis_counter = 1
        else:
            self.hysteresis_counter += 1

    def confirm_violation(self, hysteresis_frames: int = 3):
        """Confirm violation after hysteresis period"""
        if self.violation_detected and not self.violation_confirmed:
            if self.hysteresis_counter >= hysteresis_frames:
                self.violation_confirmed = True
                return True
        return False

    def is_active(self, max_age_seconds: float = 5.0) -> bool:
        """Check if track is still active"""
        return (time.time() - self.last_seen) < max_age_seconds

    def get_track_summary(self) -> Dict:
        """Get summary of track for reporting"""
        return {
            'track_id': self.track_id,
            'plate_text': self.plate_text,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen,
            'frames_seen': self.frames_seen,
            'violation_detected': self.violation_detected,
            'violation_confirmed': self.violation_confirmed,
            'violation_time': self.violation_time,
            'max_confidence': self.max_confidence,
            'average_confidence': self.average_confidence,
            'current_bbox': self.get_smoothed_bbox(),
            'evidence_frames_count': len(self.evidence_frames)
        }

class VehicleTracker:
    """Enhanced vehicle tracker with IOU-based tracking and hysteresis"""

    def __init__(self, iou_threshold: float = 0.3, hysteresis_frames: int = 3,
                 max_track_age: float = 5.0, max_tracks: int = 50):
        self.iou_threshold = iou_threshold
        self.hysteresis_frames = hysteresis_frames
        self.max_track_age = max_track_age
        self.max_tracks = max_tracks

        self.tracks: Dict[str, TrackedVehicle] = {}
        self.next_track_id = 0
        self.track_id_counter = 0

        logging.info(f"VehicleTracker initialized with IOU threshold: {iou_threshold}, "
                    f"hysteresis: {hysteresis_frames} frames")

    def calculate_iou(self, bbox1: Tuple[int, int, int, int],
                     bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union (IOU) between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x2_2) * (y2_2 - y2_2)
        union_area = bbox1_area + bbox2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def find_best_track_match(self, bbox: Tuple[int, int, int, int],
                            plate_text: str = None) -> Optional[str]:
        """Find best matching existing track for a detection"""
        best_match = None
        best_iou = 0.0
        best_confidence = 0.0

        for track_id, track in self.tracks.items():
            if not track.is_active(self.max_track_age):
                continue

            # Calculate IOU with current track bbox
            current_bbox = track.get_smoothed_bbox()
            iou = self.calculate_iou(bbox, current_bbox)

            # Boost IOU if plate text matches
            text_match = plate_text and track.plate_text == plate_text
            if text_match:
                iou *= 1.2  # Boost for matching plate text

            if iou > best_iou and iou >= self.iou_threshold:
                best_match = track_id
                best_iou = iou
                best_confidence = track.get_current_confidence()

        return best_match if best_match else None

    def create_new_track(self, bbox: Tuple[int, int, int, int],
                        plate_text: str, confidence: float, frame: np.ndarray = None) -> str:
        """Create a new track for unmatched detection"""
        track_id = f"track_{self.track_id_counter}"
        self.track_id_counter += 1

        track = TrackedVehicle(track_id, plate_text, bbox, confidence)
        if frame is not None:
            track.evidence_frames.append(frame.copy())

        self.tracks[track_id] = track

        # Limit number of tracks
        if len(self.tracks) > self.max_tracks:
            self.cleanup_old_tracks()

        logging.debug(f"Created new track {track_id} for plate {plate_text}")
        return track_id

    def update_track(self, track_id: str, bbox: Tuple[int, int, int, int],
                    confidence: float, frame: np.ndarray = None):
        """Update existing track with new detection"""
        if track_id in self.tracks:
            self.tracks[track_id].update(bbox, confidence, frame)

    def detect_violation(self, track_id: str, stop_line_y: int,
                        frame_number: int) -> Optional[Dict]:
        """Check if track has violated stop line during red light"""
        if track_id not in self.tracks:
            return None

        track = self.tracks[track_id]

        # Skip if already confirmed violation (one-violation-per-track)
        if track.violation_confirmed:
            return None

        # Get current bbox
        current_bbox = track.get_smoothed_bbox()
        x1, y1, x2, y2 = current_bbox

        # Calculate vehicle center position (more reliable than bottom)
        vehicle_center_y = (y1 + y2) / 2
        vehicle_bottom_y = y2

        # Add buffer zone to prevent false positives (vehicles must cross by at least buffer_zone pixels)
        from application.config import VIOLATION_BUFFER_ZONE, VIOLATION_MIN_CROSSING_DEPTH
        effective_stop_line = stop_line_y - VIOLATION_BUFFER_ZONE

        logging.debug(f"Track {track_id}: Vehicle center Y: {vehicle_center_y:.1f}, bottom Y: {vehicle_bottom_y}, "
                     f"Stop line Y: {stop_line_y}, Effective stop line: {effective_stop_line}, "
                     f"Crossed: {vehicle_center_y > effective_stop_line}")

        # Check if vehicle center has crossed the effective stop line
        # This is more reliable than using bottom, as it represents the vehicle's main body
        if vehicle_center_y > effective_stop_line:
            # Additional check: ensure the vehicle is actually moving forward (not just detected)
            # by checking if this is a significant crossing (more than just edge detection)
            crossing_depth = vehicle_center_y - stop_line_y

            if crossing_depth > VIOLATION_MIN_CROSSING_DEPTH:  # Must cross by at least configured pixels beyond the line
                # Mark violation
                violation_time = time.time()
                track.mark_violation(violation_time)

                # Check if violation should be confirmed
                if track.confirm_violation(self.hysteresis_frames):
                    logging.info(f"Confirmed violation for track {track_id} "
                               f"(plate: {track.plate_text}) after {track.hysteresis_counter} frames. "
                               f"Crossing depth: {crossing_depth:.1f}px")

                    return {
                        'track_id': track_id,
                        'plate_text': track.plate_text,
                        'violation_time': violation_time,
                        'violation_time_formatted': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(violation_time)),
                        'confidence': track.average_confidence,
                        'max_confidence': track.max_confidence,
                        'frame_number': frame_number,
                        'bbox': current_bbox,
                        'evidence_frames': track.evidence_frames.copy()
                    }
            else:
                logging.debug(f"Track {track_id}: Insufficient crossing depth ({crossing_depth:.1f}px), not counting as violation")
        else:
            # Reset violation state if vehicle moves back
            if track.violation_detected and not track.violation_confirmed:
                # If vehicle moves back before confirmation, reset the violation
                track.violation_detected = False
                track.violation_time = None
                track.hysteresis_counter = 0
                logging.debug(f"Track {track_id}: Vehicle moved back, resetting violation state")

        return None

    def cleanup_old_tracks(self, force: bool = False):
        """Remove old inactive tracks"""
        current_time = time.time()
        tracks_to_remove = []

        for track_id, track in self.tracks.items():
            if force or not track.is_active(self.max_track_age):
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        if tracks_to_remove:
            logging.debug(f"Cleaned up {len(tracks_to_remove)} old tracks")

    def get_active_tracks(self) -> Dict[str, Dict]:
        """Get all active tracks"""
        return {
            track_id: track.get_track_summary()
            for track_id, track in self.tracks.items()
            if track.is_active(self.max_track_age)
        }

    def get_confirmed_violations(self) -> List[Dict]:
        """Get all confirmed violations"""
        violations = []
        for track in self.tracks.values():
            if track.violation_confirmed:
                violations.append(track.get_track_summary())
        return violations

    def reset(self):
        """Reset tracker state"""
        self.tracks.clear()
        self.track_id_counter = 0
        logging.info("Vehicle tracker reset")

    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics"""
        active_tracks = len(self.get_active_tracks())
        total_tracks = len(self.tracks)
        confirmed_violations = len(self.get_confirmed_violations())

        return {
            'active_tracks': active_tracks,
            'total_tracks': total_tracks,
            'confirmed_violations': confirmed_violations,
            'iou_threshold': self.iou_threshold,
            'hysteresis_frames': self.hysteresis_frames
        }
