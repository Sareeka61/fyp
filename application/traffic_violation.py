import cv2
import numpy as np
import logging
import time
import datetime
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

def format_nepali_time(timestamp: float) -> str:
    """Convert Unix timestamp to Nepali standard time format (yyyy-mm-dd hh:mm:ss)"""
    dt = datetime.datetime.fromtimestamp(timestamp)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

@dataclass
class TrafficLightState:
    """Represents the state of a traffic light"""
    is_red: bool
    is_green: bool
    confidence: float
    timestamp: float
    cycle_position: float  # 0.0 to 1.0 representing position in light cycle

@dataclass
class VehicleTrack:
    """Tracks a vehicle across multiple frames"""
    plate_bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    plate_text: str
    confidence: float
    first_seen: float
    last_seen: float
    frames_seen: int
    violation_detected: bool
    violation_time: Optional[float]
    violation_evidence: List[np.ndarray]  # Store frames showing violation

@dataclass
class ViolationEvent:
    """Records a red light violation event"""
    plate_text: str
    violation_time: float  # Unix timestamp
    violation_time_formatted: str  # Formatted as yyyy-mm-dd hh:mm:ss
    confidence: float
    evidence_frames: List[np.ndarray]
    frame_number: int

class TrafficViolationDetector:
    """Main class for detecting red light violations with simulated traffic light"""
    
    def __init__(self):
        # Start with green light to ensure cycling begins properly
        self.traffic_light_state = TrafficLightState(False, True, 1.0, time.time(), 0.0)
        self.vehicle_tracks: Dict[str, VehicleTrack] = {}  # Track by plate text
        self.violation_events: List[ViolationEvent] = []
        self.stop_line_y = None  # Y-coordinate of stop line (to be configured)
        self.track_history = deque(maxlen=100)  # Keep recent tracking data
        # Start slightly in the past to ensure we begin in green phase
        self.start_time = time.time() - 0.1
        self.current_frame_number = 0
        self.red_light_start_time = None
        self.green_light_duration = 2.0  # 2 seconds green light
        self.red_light_duration = 4.0    # 4 seconds red light
        self.total_cycle_time = self.green_light_duration + self.red_light_duration
        
    def simulate_traffic_light(self, frame_number: int, fps: float = 3.0) -> TrafficLightState:
        """
        Simulate traffic light cycle based on time and frame number
        Returns: TrafficLightState object with simulated light state
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Calculate position in cycle (0.0 to 1.0)
        cycle_position = (elapsed_time % self.total_cycle_time) / self.total_cycle_time

        # Determine light state based on cycle position
        green_phase_end = self.green_light_duration / self.total_cycle_time

        if cycle_position < green_phase_end:
            # Green light phase
            is_green = True
            is_red = False
            self.red_light_start_time = None
            confidence = 1.0
        else:
            # Red light phase
            is_green = False
            is_red = True
            if self.red_light_start_time is None:
                self.red_light_start_time = current_time

        # Calculate confidence based on how long we've been in red phase
        red_elapsed = elapsed_time - self.green_light_duration
        confidence = min(1.0, red_elapsed / 2.0)  # Ramp up confidence over 2 seconds

        # Update the traffic_light_state attribute to reflect current state
        self.traffic_light_state = TrafficLightState(is_red, is_green, confidence, current_time, cycle_position)

        logging.debug(f"Frame {frame_number}: Traffic light SIMULATION - Elapsed: {elapsed_time:.1f}s, Cycle: {cycle_position:.2f}, "
                     f"RED: {is_red}, GREEN: {is_green}, Confidence: {confidence:.2f}")

        return self.traffic_light_state

    def simulate_traffic_light_by_frame(self, frame_number: int, fps: float = 3.0) -> TrafficLightState:
        """
        Simulate traffic light cycle based on frame number for more predictable behavior
        Returns: TrafficLightState object with simulated light state
        """
        # Calculate cycle based on frame number
        frames_per_green = int(self.green_light_duration * fps)  # ~9 frames for 3s at 3fps
        frames_per_red = int(self.red_light_duration * fps)     # ~15 frames for 5s at 3fps
        frames_per_cycle = frames_per_green + frames_per_red

        frame_in_cycle = frame_number % frames_per_cycle

        current_time = time.time()

        if frame_in_cycle < frames_per_green:
            # Green light phase
            is_green = True
            is_red = False
            confidence = 1.0
            cycle_position = frame_in_cycle / frames_per_cycle
        else:
            # Red light phase
            is_green = False
            is_red = True
            confidence = 1.0
            cycle_position = frame_in_cycle / frames_per_cycle

        logging.debug(f"Frame {frame_number}: Traffic light FRAME-SIMULATION - Frame in cycle: {frame_in_cycle}/{frames_per_cycle}, "
                     f"RED: {is_red}, GREEN: {is_green}, Cycle: {cycle_position:.2f}")

        return TrafficLightState(is_red, is_green, confidence, current_time, cycle_position)
    
    def detect_traffic_light_state(self, frame: np.ndarray, frame_number: int) -> TrafficLightState:
        """
        Detect traffic light state from video frame
        Returns: TrafficLightState object with detected light state
        """
        self.current_frame_number = frame_number

        # Force simulation instead of detection for consistent cycling
        simulated_state = self.simulate_traffic_light(frame_number)
        logging.debug(f"Frame {frame_number}: Using SIMULATION - RED: {simulated_state.is_red}, GREEN: {simulated_state.is_green}, Cycle: {simulated_state.cycle_position:.2f}")
        return simulated_state

    def detect_traffic_light_from_frame(self, frame: np.ndarray) -> Optional[TrafficLightState]:
        """
        Detect traffic light color from video frame
        Returns: TrafficLightState if detected, None if detection fails
        """
        try:
            height, width = frame.shape[:2]

            # Define traffic light ROI (top-center area of frame)
            # Adjust these coordinates based on your video setup
            roi_x1 = int(width * 0.4)  # 40% from left
            roi_y1 = int(height * 0.05)  # 5% from top
            roi_x2 = int(width * 0.6)   # 60% from left
            roi_y2 = int(height * 0.25)  # 25% from top

            # Extract ROI
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

            if roi.size == 0:
                return None

            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Define color ranges for traffic lights
            # Red color range (considering different shades of red)
            red_lower1 = np.array([0, 50, 50])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([160, 50, 50])
            red_upper2 = np.array([180, 255, 255])

            # Green color range
            green_lower = np.array([40, 50, 50])
            green_upper = np.array([80, 255, 255])

            # Yellow/Amber color range
            yellow_lower = np.array([15, 50, 50])
            yellow_upper = np.array([35, 255, 255])

            # Create masks for each color
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)

            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

            # Calculate the percentage of pixels for each color
            total_pixels = roi.shape[0] * roi.shape[1]

            red_pixels = cv2.countNonZero(red_mask)
            green_pixels = cv2.countNonZero(green_mask)
            yellow_pixels = cv2.countNonZero(yellow_mask)

            red_percentage = (red_pixels / total_pixels) * 100
            green_percentage = (green_pixels / total_pixels) * 100
            yellow_percentage = (yellow_pixels / total_pixels) * 100

            # Determine traffic light state based on dominant color
            current_time = time.time()
            confidence = 0.0

            # Thresholds for detection (adjust based on your setup)
            color_threshold = 0.5  # 0.5% of ROI must be the detected color

            if red_percentage > color_threshold:
                # Red light detected
                is_red = True
                is_green = False
                confidence = min(1.0, red_percentage / 5.0)  # Scale confidence
                logging.debug(f"RED light detected: {red_percentage:.2f}% red pixels, confidence: {confidence:.2f}")
            elif green_percentage > color_threshold:
                # Green light detected
                is_red = False
                is_green = True
                confidence = min(1.0, green_percentage / 5.0)  # Scale confidence
                logging.debug(f"GREEN light detected: {green_percentage:.2f}% green pixels, confidence: {confidence:.2f}")
            elif yellow_percentage > color_threshold:
                # Yellow/Amber light detected - treat as red for safety
                is_red = True
                is_green = False
                confidence = min(1.0, yellow_percentage / 5.0)  # Scale confidence
                logging.debug(f"YELLOW light detected: {yellow_percentage:.2f}% yellow pixels, treating as RED")
            else:
                # No clear color detected
                logging.debug(f"Frame {self.current_frame_number}: No traffic light color detected clearly: R:{red_percentage:.2f}%, G:{green_percentage:.2f}%, Y:{yellow_percentage:.2f}% (ROI: {roi_x1}-{roi_x2}, {roi_y1}-{roi_y2})")
                return None

            # Calculate cycle position (estimate based on current state)
            cycle_position = 0.5 if is_red else 0.0  # Rough estimate

            return TrafficLightState(
                is_red=is_red,
                is_green=is_green,
                confidence=confidence,
                timestamp=current_time,
                cycle_position=cycle_position
            )

        except Exception as e:
            logging.error(f"Error detecting traffic light from frame: {e}")
            return None
    
    def set_stop_line(self, y_coordinate: int):
        """Set the stop line Y-coordinate for violation detection"""
        self.stop_line_y = y_coordinate
        logging.info(f"Stop line set at Y-coordinate: {y_coordinate}")
    
    def detect_violation(self, frame: np.ndarray, frame_number: int, 
                        plate_bbox: Tuple[int, int, int, int], 
                        plate_text: str, confidence: float) -> Optional[ViolationEvent]:
        """
        Detect if a vehicle is violating red light rules
        Returns: ViolationEvent if violation detected, None otherwise
        """
        if self.stop_line_y is None:
            logging.warning("Stop line not configured. Cannot detect violations.")
            return None
        
        if not self.traffic_light_state.is_red:
            return None  # Only check violations during red light
        
        x1, y1, x2, y2 = plate_bbox
        vehicle_bottom_y = y2  # Bottom of vehicle bounding box
        
        # Check if vehicle has crossed stop line
        if vehicle_bottom_y > self.stop_line_y:
            # This is a potential violation
            current_time = time.time()
            
            # Check if we're already tracking this vehicle
            track_key = f"{plate_text}_{frame_number}"
            
            if track_key not in self.vehicle_tracks:
                # Start new track
                self.vehicle_tracks[track_key] = VehicleTrack(
                    plate_bbox=plate_bbox,
                    plate_text=plate_text,
                    confidence=confidence,
                    first_seen=current_time,
                    last_seen=current_time,
                    frames_seen=1,
                    violation_detected=True,
                    violation_time=current_time,
                    violation_evidence=[frame.copy()]
                )
                logging.info(f"Potential violation detected: {plate_text}")
            else:
                # Update existing track
                track = self.vehicle_tracks[track_key]
                track.last_seen = current_time
                track.frames_seen += 1
                track.violation_evidence.append(frame.copy())
            
            # Create violation event
            violation = ViolationEvent(
                plate_text=plate_text,
                violation_time=current_time,
                violation_time_formatted=format_nepali_time(current_time),
                confidence=confidence,
                evidence_frames=[frame.copy()],
                frame_number=frame_number
            )
            
            self.violation_events.append(violation)
            return violation
        
        return None
    
    def draw_detection_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Draw detection overlays on the frame for visualization"""
        overlay = frame.copy()
        
        # Draw stop line if configured
        if self.stop_line_y is not None:
            cv2.line(overlay, (0, self.stop_line_y), 
                     (frame.shape[1], self.stop_line_y), 
                     (0, 0, 255), 2)  # Red line
        
        # Draw traffic light status
        light_status = "RED" if self.traffic_light_state.is_red else "GREEN" if self.traffic_light_state.is_green else "UNKNOWN"
        light_color = (0, 0, 255) if self.traffic_light_state.is_red else (0, 255, 0) if self.traffic_light_state.is_green else (255, 255, 255)
        
        cv2.putText(overlay, f"Traffic Light: {light_status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, light_color, 2)
        
        # Draw violation count
        cv2.putText(overlay, f"Violations: {len(self.violation_events)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return overlay
    
    def clear_violations(self):
        """Clear all violation records"""
        self.violation_events.clear()
        self.vehicle_tracks.clear()
        logging.info("Cleared all violation records")
    
    def get_violation_summary(self) -> Dict:
        """Get summary of all detected violations"""
        return {
            "total_violations": len(self.violation_events),
            "violations": [
                {
                    "plate_text": violation.plate_text,
                    "violation_time": violation.violation_time_formatted,
                    "confidence": violation.confidence,
                    "frame_number": violation.frame_number
                }
                for violation in self.violation_events
            ]
        }
