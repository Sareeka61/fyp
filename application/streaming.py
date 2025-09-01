import cv2
import time
import logging
from flask import Response
from typing import Generator
import io
from queue import Empty

def generate_mjpeg_stream(job):
    """
    Generate MJPEG stream from job's frame queue
    Yields multipart JPEG frames for live preview
    """
    logging.info(f"Starting MJPEG stream for job {job.job_id}")

    while not job.stop_event.is_set() and job.status.value == 'processing':
        try:
            # Get frame from queue with timeout
            frame_data = job.frame_queue.get(timeout=1.0)

            if frame_data and 'frame' in frame_data:
                frame = frame_data['frame']

                # Encode frame as JPEG
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                _, encoded_frame = cv2.imencode('.jpg', frame, encode_param)

                # Convert to bytes
                frame_bytes = encoded_frame.tobytes()

                # Yield multipart response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        except Empty:
            # No frame available, send a placeholder or wait
            time.sleep(0.1)
            continue
        except Exception as e:
            logging.error(f"Error in MJPEG stream for job {job.job_id}: {e}")
            break

    logging.info(f"Ended MJPEG stream for job {job.job_id}")

def create_mjpeg_response(job):
    """
    Create Flask Response for MJPEG streaming
    """
    return Response(
        generate_mjpeg_stream(job),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*'
        }
    )

def add_frame_to_job(job, frame, frame_number=None, overlays=None):
    """
    Add a processed frame to the job's queue for streaming
    Includes optional overlays for annotations
    """
    if job.status != job.status.PROCESSING:
        return

    # Add overlays if provided
    display_frame = frame.copy()
    if overlays:
        for overlay in overlays:
            if overlay['type'] == 'text':
                cv2.putText(display_frame, overlay['text'], overlay['position'],
                           cv2.FONT_HERSHEY_SIMPLEX, overlay['font_scale'],
                           overlay['color'], overlay['thickness'])
            elif overlay['type'] == 'rectangle':
                cv2.rectangle(display_frame, overlay['start_point'], overlay['end_point'],
                             overlay['color'], overlay['thickness'])
            elif overlay['type'] == 'line':
                cv2.line(display_frame, overlay['start_point'], overlay['end_point'],
                        overlay['color'], overlay['thickness'])
            elif overlay['type'] == 'circle':
                cv2.circle(display_frame, overlay['center'], overlay['radius'],
                          overlay['color'], overlay['thickness'])
            elif overlay['type'] == 'filled_circle':
                cv2.circle(display_frame, overlay['center'], overlay['radius'],
                          overlay['color'], -1)  # -1 for filled

    # Resize frame if needed for streaming
    target_width = 1280  # Configurable
    target_height = 720  # Configurable

    if display_frame.shape[1] != target_width or display_frame.shape[0] != target_height:
        display_frame = cv2.resize(display_frame, (target_width, target_height))

    frame_data = {
        'frame': display_frame,
        'frame_number': frame_number,
        'timestamp': time.time()
    }

    job.add_frame(frame_data)

def create_overlay_annotations(traffic_detector, frame_number, violations_count, frame_width=1280, frame_height=720, detected_plates=None):
    """
    Create overlay annotations for the frame including detected plates
    """
    overlays = []

    # Traffic light visual display with enhanced information
    if hasattr(traffic_detector, 'traffic_light_state') and traffic_detector.traffic_light_state:
        is_red = traffic_detector.traffic_light_state.is_red
        is_green = traffic_detector.traffic_light_state.is_green
        confidence = getattr(traffic_detector.traffic_light_state, 'confidence', 0.0)

        # Draw traffic light circles in top-right corner
        light_center_x = frame_width - 80  # 80px from right edge
        red_light_center = (light_center_x, 60)
        vertical_spacing = 2 * 50 + 15  # 2 * light_radius + margin
        green_light_center = (light_center_x, 60 + vertical_spacing)
        light_radius = 50

        # Draw traffic light housing (black rectangle)
        housing_top = 20
        housing_bottom = 60 + vertical_spacing + light_radius + 10
        housing_left = light_center_x - light_radius - 10
        housing_right = light_center_x + light_radius + 10

        overlays.append({
            'type': 'rectangle',
            'start_point': (housing_left, housing_top),
            'end_point': (housing_right, housing_bottom),
            'color': (0, 0, 0),  # Black housing
            'thickness': -1  # Filled
        })

        # Draw traffic light border
        overlays.append({
            'type': 'rectangle',
            'start_point': (housing_left, housing_top),
            'end_point': (housing_right, housing_bottom),
            'color': (255, 255, 255),  # White border
            'thickness': 2
        })

        if is_red:
            # Red light active (filled and bright)
            overlays.append({
                'type': 'filled_circle',
                'center': red_light_center,
                'radius': light_radius,
                'color': (0, 0, 255)  # Bright red
            })
            # Red light border
            overlays.append({
                'type': 'circle',
                'center': red_light_center,
                'radius': light_radius,
                'color': (255, 255, 255),  # White border
                'thickness': 2
            })
            # Green light inactive (dimmed)
            overlays.append({
                'type': 'filled_circle',
                'center': green_light_center,
                'radius': light_radius,
                'color': (0, 50, 0)  # Dim green
            })
        elif is_green:
            # Green light active (filled and bright)
            overlays.append({
                'type': 'filled_circle',
                'center': green_light_center,
                'radius': light_radius,
                'color': (0, 255, 0)  # Bright green
            })
            # Green light border
            overlays.append({
                'type': 'circle',
                'center': green_light_center,
                'radius': light_radius,
                'color': (255, 255, 255),  # White border
                'thickness': 2
            })
            # Red light inactive (dimmed)
            overlays.append({
                'type': 'filled_circle',
                'center': red_light_center,
                'radius': light_radius,
                'color': (50, 0, 0)  # Dim red
            })
        else:
            # Unknown state - both dimmed
            overlays.append({
                'type': 'filled_circle',
                'center': red_light_center,
                'radius': light_radius,
                'color': (50, 0, 0)  # Dim red
            })
            overlays.append({
                'type': 'filled_circle',
                'center': green_light_center,
                'radius': light_radius,
                'color': (0, 50, 0)  # Dim green
            })

        # Add traffic light status text
        light_status = "RED LIGHT" if is_red else "GREEN LIGHT" if is_green else "UNKNOWN"
        status_color = (0, 0, 255) if is_red else (0, 255, 0) if is_green else (255, 255, 255)

        overlays.append({
            'type': 'text',
            'text': f"TRAFFIC LIGHT: {light_status}",
            'position': (housing_left - 200, housing_top + 30),
            'font_scale': 0.8,
            'color': status_color,
            'thickness': 3
        })

        # Add confidence if available
        if confidence > 0:
            overlays.append({
                'type': 'text',
                'text': f"Confidence: {confidence:.2f}",
                'position': (housing_left - 200, housing_top + 60),
                'font_scale': 0.6,
                'color': (255, 255, 255),
                'thickness': 2
            })

        logging.debug(f"Overlay: Traffic light - Status: {light_status}, Confidence: {confidence:.2f}")
    else:
        logging.warning("Overlay: Traffic light state not available")

    # Violations count
    overlays.append({
        'type': 'text',
        'text': f"Violations: {violations_count}",
        'position': (10, 60),
        'font_scale': 0.7,
        'color': (0, 0, 255),
        'thickness': 2
    })

    # Frame number and processing info
    overlays.append({
        'type': 'text',
        'text': f"Frame: {frame_number}",
        'position': (10, 90),
        'font_scale': 0.7,
        'color': (255, 255, 255),
        'thickness': 2
    })

    # Detected plates count
    plates_count = len(detected_plates) if detected_plates else 0
    overlays.append({
        'type': 'text',
        'text': f"Plates Detected: {plates_count}",
        'position': (10, 120),
        'font_scale': 0.7,
        'color': (255, 255, 0),
        'thickness': 2
    })

    # Processing status
    overlays.append({
        'type': 'text',
        'text': "STATUS: ACTIVE PROCESSING",
        'position': (10, 150),
        'font_scale': 0.7,
        'color': (0, 255, 255),
        'thickness': 2
    })

    # System information
    overlays.append({
        'type': 'text',
        'text': "SYSTEM: RED LIGHT VIOLATION DETECTOR",
        'position': (10, frame_height - 60),
        'font_scale': 0.6,
        'color': (200, 200, 200),
        'thickness': 2
    })

    overlays.append({
        'type': 'text',
        'text': "Real-time + Traffic Light Detection",
        'position': (10, frame_height - 35),
        'font_scale': 0.5,
        'color': (150, 150, 150),
        'thickness': 1
    })

    overlays.append({
        'type': 'text',
        'text': "Press 'Q' to stop processing",
        'position': (10, frame_height - 10),
        'font_scale': 0.5,
        'color': (100, 100, 100),
        'thickness': 1
    })

    # Stop line with enhanced visibility
    if hasattr(traffic_detector, 'stop_line_y') and traffic_detector.stop_line_y:
        stop_line_y = traffic_detector.stop_line_y

        # Draw main stop line
        overlays.append({
            'type': 'line',
            'start_point': (0, stop_line_y),
            'end_point': (frame_width, stop_line_y),
            'color': (0, 0, 255),  # Red line
            'thickness': 4
        })

        # Draw dashed line effect with small rectangles
        dash_width = 20
        dash_gap = 10
        for x in range(0, frame_width, dash_width + dash_gap):
            overlays.append({
                'type': 'rectangle',
                'start_point': (x, stop_line_y - 2),
                'end_point': (x + dash_width, stop_line_y + 2),
                'color': (255, 255, 255),  # White dashes
                'thickness': -1  # Filled
            })

        # Add stop line label
        label_position = (10, stop_line_y - 15)
        overlays.append({
            'type': 'text',
            'text': "STOP LINE - DO NOT CROSS WHEN RED LIGHT",
            'position': label_position,
            'font_scale': 0.8,
            'color': (0, 0, 255),
            'thickness': 3
        })

        # Add stop line Y coordinate for debugging
        coord_position = (10, stop_line_y + 25)
        overlays.append({
            'type': 'text',
            'text': f"Y={stop_line_y}",
            'position': coord_position,
            'font_scale': 0.6,
            'color': (255, 255, 255),
            'thickness': 2
        })

        logging.debug(f"Overlay: Enhanced stop line at Y={stop_line_y}")

    # Add detected plates overlays with enhanced information
    if detected_plates:
        for i, plate in enumerate(detected_plates):
            if 'coordinates' in plate and 'text' in plate:
                coords = plate['coordinates']
                text = plate['text']
                confidence = plate.get('confidence', 0.0)

                # Determine box color based on violation status
                box_color = (0, 0, 255) if plate.get('violation', False) else (0, 255, 0)  # Red for violation, green for no violation

                # Draw bounding box with thicker line for violations
                thickness = 4 if plate.get('violation', False) else 2
                overlays.append({
                    'type': 'rectangle',
                    'start_point': (coords['x1'], coords['y1']),
                    'end_point': (coords['x2'], coords['y2']),
                    'color': box_color,
                    'thickness': thickness
                })

                # Draw plate text with violation status
                violation_text = "VIOLATION!" if plate.get('violation', False) else ""
                text_position = (coords['x1'], coords['y1'] - 35)
                overlays.append({
                    'type': 'text',
                    'text': f"Plate: {text} ({confidence:.2f})",
                    'position': text_position,
                    'font_scale': 0.7,
                    'color': box_color,
                    'thickness': 2
                })

                # Draw violation text if applicable
                if violation_text:
                    violation_position = (coords['x1'], coords['y1'] - 10)
                    overlays.append({
                        'type': 'text',
                        'text': violation_text,
                        'position': violation_position,
                        'font_scale': 0.8,
                        'color': (0, 0, 255),
                        'thickness': 3
                    })

                logging.debug(f"Overlay: Plate {i+1} - '{text}' at ({coords['x1']},{coords['y1']}) - Violation: {plate.get('violation', False)}")

    return overlays
