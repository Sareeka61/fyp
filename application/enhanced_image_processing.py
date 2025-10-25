"""
Enhanced image processing pipeline with real-time stats emission
Integrates with SocketIO and stats manager for live updates
"""
import cv2
import os
import logging
import time
import base64
from typing import Optional, Dict, Any

from application import config
from application.utils import to_base64
from application.character_processing import deskew_plate, process_and_order_characters, create_digital_plate
from application.traffic_violation import TrafficViolationDetector
from application.vehicle_tracker import VehicleTracker
from application.file_outputs import FileOutputGenerator
from application.stats_manager import stats_manager, ViolationEvent
from application.socketio_manager import socketio_manager

def process_frame_with_stats(frame, frame_number, filename_prefix, plate_model, seg_model, recog_model, device, ocr_font, job_id):
    """
    Enhanced frame processing with real-time stats emission
    """
    if plate_model is None or seg_model is None or recog_model is None:
        logging.error(f"One or more models are not loaded. Cannot process frame {frame_number} from {filename_prefix}.")
        return []
    if frame is None or frame.size == 0:
         logging.error(f"Received empty frame {frame_number} for processing.")
         return []

    frame_results_list = []
    h_frame, w_frame = frame.shape[:2]
    start_time = time.time()
    
    # Update frame processed count
    stats_manager.update_frame_processed(job_id)
    
    logging.debug(f"Processing Frame: {frame_number} from '{filename_prefix}' ({w_frame}x{h_frame})")

    try:
        plate_results = plate_model.predict(frame, verbose=False, conf=config.PLATE_DETECT_CONF)
    except Exception as e:
        logging.error(f"Plate detection failed on frame {frame_number} ({filename_prefix}): {e}", exc_info=True)
        return []

    if not plate_results or not plate_results[0].boxes:
        logging.debug(f"No plates detected in frame {frame_number} ({filename_prefix}).")
        return []

    detected_boxes = plate_results[0].boxes
    logging.debug(f"Frame {frame_number}: Found {len(detected_boxes)} potential plate(s).")

    for i, plate_box in enumerate(detected_boxes):
        plate_start_time = time.time()
        plate_info = {}

        try:
            conf = float(plate_box.conf[0])
            x1, y1, x2, y2 = map(int, plate_box.xyxy[0].tolist())
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_frame, x2), min(h_frame, y2)

            if x1 >= x2 or y1 >= y2:
                logging.warning(f"Skipping invalid plate box {i} in frame {frame_number}: [{x1},{y1},{x2},{y2}]")
                continue

            plate_img = frame[y1:y2, x1:x2].copy()
            plate_info['original_plate'] = to_base64(plate_img)
            plate_info['plate_coordinates'] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
            plate_info['plate_dimensions'] = {'width': plate_img.shape[1], 'height': plate_img.shape[0]}
            plate_info['confidence'] = conf
            plate_info['frame_number'] = frame_number
            plate_info['plate_index'] = i
            plate_info['filename'] = filename_prefix

            deskewed_plate = deskew_plate(plate_img)
            if deskewed_plate is None or deskewed_plate.size == 0:
                logging.warning(f"Deskewing returned empty image for plate {i}, frame {frame_number}. Using original.")
                deskewed_plate = plate_img
            plate_info['deskewed_plate'] = to_base64(deskewed_plate)
            plate_info['deskewed_dimensions'] = {'width': deskewed_plate.shape[1], 'height': deskewed_plate.shape[0]}

            char_seg_results = None
            ordered_characters = []
            final_text = ""
            digital_plate_str = to_base64(None)

            if deskewed_plate is not None and deskewed_plate.shape[0] > 5 and deskewed_plate.shape[1] > 5:
                 try:
                     char_seg_results = seg_model.predict(deskewed_plate, verbose=False, conf=config.CHAR_SEG_CONF)
                 except Exception as e:
                     logging.error(f"Char segmentation failed for plate {i}, frame {frame_number}: {e}", exc_info=True)

                 if char_seg_results:
                     try:
                         ordered_characters, final_text = process_and_order_characters(
                             deskewed_plate, char_seg_results, recog_model, device
                         )
                         plate_info['characters'] = ordered_characters
                         plate_info['final_text'] = final_text
                         logging.info(f"Frame {frame_number}, Plate {i}: Recognized Text='{final_text}' (Plate Conf:{conf:.2f})")
                     except Exception as e:
                         logging.error(f"Character processing/ordering failed for plate {i}, frame {frame_number}: {e}", exc_info=True)
                         plate_info['characters'] = []
                         plate_info['final_text'] = "[OCR Error]"

            try:
                digital_plate_pil = create_digital_plate(deskewed_plate.shape[:2], plate_info.get('characters', []), ocr_font)
                digital_plate_str = to_base64(digital_plate_pil)
            except Exception as e:
                logging.error(f"Digital plate creation failed for plate {i}, frame {frame_number}: {e}", exc_info=True)

            plate_info['digital_plate'] = digital_plate_str
            frame_results_list.append(plate_info)
            
            plate_end_time = time.time()
            logging.debug(f"Plate {i} processed in {plate_end_time - plate_start_time:.3f} seconds.")

        except Exception as plate_proc_err:
            logging.error(f"Error processing detected plate {i} in frame {frame_number} ({filename_prefix}): {plate_proc_err}", exc_info=True)
            frame_results_list.append({
                'error': f"Failed to process plate {i}",
                'frame_number': frame_number,
                'plate_index': i,
                'filename': filename_prefix,
                'original_plate': to_base64(plate_img) if 'plate_img' in locals() else to_base64(None),
            })
            continue

    end_time = time.time()
    logging.debug(f"Frame {frame_number} processing took {end_time - start_time:.3f} seconds. Found {len(frame_results_list)} valid plates.")
    return frame_results_list

def process_video_for_job_enhanced(job, plate_model, seg_model, recog_model, device, ocr_font):
    """
    Enhanced video processing with real-time stats and SocketIO emission
    """
    from application.streaming import add_frame_to_job, create_overlay_annotations

    file_path = job.video_path
    filename = os.path.basename(file_path)
    job_id = job.job_id

    logging.info(f"Starting enhanced job processing for {filename} (Job: {job_id})")

    # Initialize stats for this job
    stats_manager.create_job_stats(job_id)
    
    # Emit job started event
    socketio_manager.emit_job_started(job_id)

    # Initialize components
    violation_detector = TrafficViolationDetector()
    violation_detector.set_stop_line(config.VIRTUAL_LINE_Y_COORDINATE)

    vehicle_tracker = VehicleTracker(
        iou_threshold=config.IOU_THRESHOLD,
        hysteresis_frames=config.HYSTERESIS_FRAMES,
        max_track_age=config.MAX_TRACK_AGE,
        max_tracks=config.MAX_ACTIVE_TRACKS
    )

    file_generator = FileOutputGenerator(job.output_dir)

    # Store frame annotations for output generation
    frame_annotations = []
    confirmed_violations = []

    cap = None
    try:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {file_path}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            logging.warning(f"Could not determine video frame rate for {filename}. Using default skip value: {config.VIDEO_FRAME_SKIP}")
            frame_skip = config.VIDEO_FRAME_SKIP
        else:
            frame_skip = max(1, int(fps / config.TARGET_FPS))
            logging.info(f"Video {filename}: {fps:.1f} FPS detected. Processing every {frame_skip} frames for ~{config.TARGET_FPS} FPS")

        job.update_progress(0, total_frames)

        frame_number = 0
        processed_count = 0  # Initialize processed frame counter

        # Use calculated frame_skip based on video FPS for consistent timing
        # Don't override with hardcoded value
        stats_emit_interval = 0.5  # Emit stats every 0.5 seconds (less frequent for performance)
        last_stats_emit = 0
        preview_emit_interval = 0.3  # Emit preview frames every 0.3 seconds (less frequent)
        last_preview_emit = 0

        while not job.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logging.info(f"End of video {filename}.")
                break

            if frame_number % frame_skip == 0:
                try:
                    # Update traffic light state for each frame using frame-based simulation for consistency
                    violation_detector.traffic_light_state = violation_detector.simulate_traffic_light_by_frame(frame_number, fps)
                    logging.debug(f"Frame {frame_number}: Traffic light state - RED: {violation_detector.traffic_light_state.is_red}, GREEN: {violation_detector.traffic_light_state.is_green}")

                    # Process frame with stats tracking
                    frame_results = process_frame_with_stats(
                        frame, frame_number, filename, plate_model, seg_model, 
                        recog_model, device, ocr_font, job_id
                    )

                    # Store frame results in job for later display
                    job.results.extend(frame_results)

                    # Update vehicle tracks and check for violations
                    current_frame_violations = []

                    for plate_info in frame_results:
                        if 'plate_coordinates' in plate_info:
                            coords = plate_info['plate_coordinates']
                            bbox = (coords['x1'], coords['y1'], coords['x2'], coords['y2'])
                            plate_text = plate_info.get('final_text', '')
                            confidence = plate_info.get('confidence', 0.0)

                            # Find or create track
                            track_id = vehicle_tracker.find_best_track_match(bbox, plate_text)
                            if track_id is None:
                                track_id = vehicle_tracker.create_new_track(bbox, plate_text, confidence, frame)
                                logging.debug(f"Frame {frame_number}: Created new track {track_id} for plate '{plate_text}'")
                            else:
                                vehicle_tracker.update_track(track_id, bbox, confidence, frame)
                                logging.debug(f"Frame {frame_number}: Updated track {track_id} for plate '{plate_text}'")

                            # Check traffic light state and violation
                            if violation_detector.traffic_light_state.is_red:
                                violation = vehicle_tracker.detect_violation(track_id, config.VIRTUAL_LINE_Y_COORDINATE, frame_number)
                                if violation:
                                    logging.info(f"VIOLATION DETECTED: Frame {frame_number}, Track {track_id}, Plate: {plate_text}")
                                    current_frame_violations.append(violation)
                                    confirmed_violations.append(violation)

                                    # Create violation event
                                    violation_event = ViolationEvent(
                                        job_id=job_id,
                                        plate=plate_text,
                                        confidence=confidence,
                                        bbox=[coords['x1'], coords['y1'], coords['x2'], coords['y2']],
                                        frame_number=frame_number
                                    )
                                    
                                    # Add to stats manager
                                    stats_manager.add_violation(job_id, violation_event)
                                    
                                    # Emit violation event via SocketIO
                                    socketio_manager.emit_violation(job_id, violation_event.to_dict())

                                    # Mark this plate as having a violation for display
                                    plate_info['violation'] = True
                                    plate_info['violation_time'] = violation['violation_time_formatted']
                                else:
                                    plate_info['violation'] = False
                            else:
                                plate_info['violation'] = False

                    # Create overlay annotations
                    detected_plates = []
                    for plate_info in frame_results:
                        if 'plate_coordinates' in plate_info:
                            coords = plate_info['plate_coordinates']
                            detected_plates.append({
                                'coordinates': {
                                    'x1': coords['x1'],
                                    'y1': coords['y1'],
                                    'x2': coords['x2'],
                                    'y2': coords['y2']
                                },
                                'text': plate_info.get('final_text', ''),
                                'confidence': plate_info.get('confidence', 0.0),
                                'violation': plate_info.get('violation', False),
                                'violation_time': plate_info.get('violation_time', '')
                            })

                    overlays = create_overlay_annotations(violation_detector, frame_number, len(confirmed_violations), frame.shape[1], frame.shape[0], detected_plates)

                    # Store frame annotations for output generation
                    frame_annotations.append({
                        'frame_number': frame_number,
                        'overlays': overlays,
                        'traffic_light': violation_detector.traffic_light_state.is_red,
                        'violations_in_frame': len(current_frame_violations),
                        'frame_image': frame.copy()
                    })

                    # Add frame to streaming queue (throttled for performance)
                    current_time = time.time()
                    if current_time - last_preview_emit >= preview_emit_interval:
                        add_frame_to_job(job, frame, frame_number, overlays)

                        # Emit preview frame via SocketIO (throttled and optimized)
                        # Resize frame for preview to reduce data size
                        preview_frame = cv2.resize(frame, (640, 360))  # Smaller preview for better performance
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                        _, encoded_frame = cv2.imencode('.jpg', preview_frame, encode_param)
                        frame_b64 = base64.b64encode(encoded_frame).decode('utf-8')
                        socketio_manager.emit_job_preview(job_id, f"data:image/jpeg;base64,{frame_b64}")
                        last_preview_emit = current_time

                    processed_count += 1
                    job.update_progress(processed_count, total_frames // frame_skip)

                    # Emit stats update via SocketIO (throttled) - only every few frames to reduce load
                    current_time = time.time()
                    if current_time - last_stats_emit >= stats_emit_interval:
                        stats = stats_manager.get_stats(job_id)
                        if stats:
                            socketio_manager.emit_job_stats(job_id, stats.to_dict())
                        last_stats_emit = current_time

                except Exception as e:
                    logging.error(f"Error processing frame {frame_number} in job {job.job_id}: {e}", exc_info=True)
                    # Continue processing despite frame errors
                    continue

            frame_number += 1

            # Small delay to prevent overwhelming the system
            time.sleep(0.01)

        # Generate file outputs
        logging.info(f"Generating file outputs for job {job.job_id}")
        metadata = {
            'filename': filename,
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'total_violations': len(confirmed_violations),
            'processing_timestamp': time.time()
        }

        outputs = file_generator.generate_all_outputs(file_path, confirmed_violations, frame_annotations, metadata, job.results)

        # Store frame snapshots in job object for display
        if 'frame_snapshots' in outputs:
            job.frame_snapshots = outputs['frame_snapshots']
            logging.info(f"Set {len(job.frame_snapshots)} frame_snapshots for job {job.job_id}")

        # Emit final completion event
        final_stats = stats_manager.get_stats(job_id)
        if final_stats:
            socketio_manager.emit_job_completed(job_id, final_stats.to_dict())

        logging.info(f"Job {job.job_id} processing completed. Processed {processed_count} frames, found {len(confirmed_violations)} violations.")

    except Exception as video_err:
        logging.error(f"Critical error during job {job.job_id} processing: {video_err}", exc_info=True)
        socketio_manager.emit_job_failed(job_id, str(video_err))
        raise
    finally:
        if cap is not None and cap.isOpened():
            logging.debug(f"Releasing video capture for job {job.job_id}")
            cap.release()

        # Cleanup trackers
        vehicle_tracker.cleanup_old_tracks(force=True)
