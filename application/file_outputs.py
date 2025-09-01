import cv2
import os
import json
import csv
import logging
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import numpy as np

class FileOutputGenerator:
    """Generate various file outputs from video processing results"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.annotated_video_path = None
        self.report_csv_path = None
        self.report_json_path = None
        self.snapshots_dir = None

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        self.snapshots_dir = os.path.join(output_dir, 'snapshots')
        os.makedirs(self.snapshots_dir, exist_ok=True)

        logging.info(f"FileOutputGenerator initialized with output dir: {output_dir}")

    def generate_annotated_video(self, video_path: str, violations: List[Dict],
                               frame_annotations: List[Dict] = None,
                               fps: float = 30.0) -> str:
        """
        Generate annotated video with overlays showing violations and detections
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {video_path}")

            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Output video path
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            self.annotated_video_path = os.path.join(self.output_dir, f"{base_name}_annotated.mp4")

            # Video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.annotated_video_path, fourcc, fps, (width, height))

            frame_number = 0
            violation_idx = 0

            logging.info(f"Generating annotated video: {self.annotated_video_path}")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Add frame annotations
                # Extract overlays for this frame
                overlays = []
                if frame_annotations:
                    for annotation in frame_annotations:
                        if annotation.get('frame_number') == frame_number:
                            overlays = annotation.get('overlays', [])
                            break
                annotated_frame = self._add_frame_annotations(frame, frame_number, violations, overlays)

                # Write frame
                out.write(annotated_frame)
                frame_number += 1

                if frame_number % 100 == 0:
                    logging.debug(f"Processed {frame_number}/{total_frames} frames for annotation")

            cap.release()
            out.release()

            logging.info(f"Annotated video generated: {self.annotated_video_path}")
            return self.annotated_video_path

        except Exception as e:
            logging.error(f"Error generating annotated video: {e}")
            return None

    def _add_frame_annotations(self, frame: np.ndarray, frame_number: int,
                             violations: List[Dict], overlays: List[Dict] = None) -> np.ndarray:
        """Add annotations to a single frame"""
        annotated = frame.copy()

        # Add overlays (traffic lights, stop lines, etc.)
        if overlays:
            for overlay in overlays:
                self._draw_overlay(annotated, overlay)

        # Add violation markers
        for violation in violations:
            if violation.get('frame_number') == frame_number:
                self._draw_violation_marker(annotated, violation)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, f"Frame: {frame_number} | Time: {timestamp}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return annotated

    def _draw_annotation(self, frame: np.ndarray, annotation: Dict):
        """Draw a single annotation on the frame"""
        ann_type = annotation.get('type', '')

        if ann_type == 'text':
            cv2.putText(frame, annotation.get('text', ''),
                       annotation.get('position', (10, 50)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       annotation.get('font_scale', 0.7),
                       annotation.get('color', (255, 255, 255)),
                       annotation.get('thickness', 2))

        elif ann_type == 'rectangle':
            cv2.rectangle(frame,
                         annotation.get('start_point', (0, 0)),
                         annotation.get('end_point', (100, 100)),
                         annotation.get('color', (0, 255, 0)),
                         annotation.get('thickness', 2))

        elif ann_type == 'line':
            cv2.line(frame,
                    annotation.get('start_point', (0, 0)),
                    annotation.get('end_point', (100, 100)),
                    annotation.get('color', (0, 0, 255)),
                    annotation.get('thickness', 2))

    def _draw_violation_marker(self, frame: np.ndarray, violation: Dict):
        """Draw violation marker on the frame"""
        bbox = violation.get('bbox')
        if bbox:
            x1, y1, x2, y2 = bbox
            # Draw red rectangle around violating vehicle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Add violation text
        plate_text = violation.get('plate_text', 'Unknown')
        cv2.putText(frame, f"VIOLATION: {plate_text}",
                   (10, frame.shape[0] - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        violation_time = violation.get('violation_time_formatted', '')
        cv2.putText(frame, f"Time: {violation_time}",
                   (10, frame.shape[0] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def generate_csv_report(self, violations: List[Dict], metadata: Dict = None) -> str:
        """Generate CSV report of violations"""
        try:
            base_name = metadata.get('filename', 'violations') if metadata else 'violations'
            self.report_csv_path = os.path.join(self.output_dir, f"{base_name}_report.csv")

            with open(self.report_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['plate_text', 'violation_time', 'confidence', 'max_confidence',
                            'frame_number', 'track_id', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for violation in violations:
                    row = {
                        'plate_text': violation.get('plate_text', ''),
                        'violation_time': violation.get('violation_time_formatted', ''),
                        'confidence': violation.get('confidence', 0.0),
                        'max_confidence': violation.get('max_confidence', 0.0),
                        'frame_number': violation.get('frame_number', 0),
                        'track_id': violation.get('track_id', ''),
                    }

                    # Add bbox coordinates
                    bbox = violation.get('bbox')
                    if bbox:
                        row.update({
                            'bbox_x1': bbox[0],
                            'bbox_y1': bbox[1],
                            'bbox_x2': bbox[2],
                            'bbox_y2': bbox[3]
                        })

                    writer.writerow(row)

            logging.info(f"CSV report generated: {self.report_csv_path}")
            return self.report_csv_path

        except Exception as e:
            logging.error(f"Error generating CSV report: {e}")
            return None

    def generate_json_report(self, violations: List[Dict], metadata: Dict = None) -> str:
        """Generate JSON report of violations and processing metadata"""
        try:
            base_name = metadata.get('filename', 'violations') if metadata else 'violations'
            self.report_json_path = os.path.join(self.output_dir, f"{base_name}_report.json")

            report_data = {
                'metadata': metadata or {},
                'processing_timestamp': datetime.now().isoformat(),
                'total_violations': len(violations),
                'violations': violations
            }

            with open(self.report_json_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(report_data, jsonfile, indent=2, ensure_ascii=False)

            logging.info(f"JSON report generated: {self.report_json_path}")
            return self.report_json_path

        except Exception as e:
            logging.error(f"Error generating JSON report: {e}")
            return None

    def generate_snapshots(self, violations: List[Dict], max_snapshots: int = 10) -> List[str]:
        """Generate snapshot images of violations"""
        try:
            snapshot_paths = []

            # Sort violations by confidence (highest first)
            sorted_violations = sorted(violations,
                                     key=lambda x: x.get('confidence', 0),
                                     reverse=True)

            for i, violation in enumerate(sorted_violations[:max_snapshots]):
                evidence_frames = violation.get('evidence_frames', [])
                if evidence_frames:
                    # Use the first evidence frame
                    frame = evidence_frames[0]

                    # Add violation annotation
                    annotated_frame = self._add_frame_annotations(frame,
                                                                violation.get('frame_number', 0),
                                                                [violation], [])

                    # Save snapshot
                    plate_text = violation.get('plate_text', 'unknown').replace(' ', '_')
                    snapshot_name = f"violation_{i+1}_{plate_text}_frame_{violation.get('frame_number', 0)}.jpg"
                    snapshot_path = os.path.join(self.snapshots_dir, snapshot_name)

                    cv2.imwrite(snapshot_path, annotated_frame)
                    snapshot_paths.append(snapshot_path)

                    logging.debug(f"Generated snapshot: {snapshot_path}")

            logging.info(f"Generated {len(snapshot_paths)} violation snapshots")
            return snapshot_paths

        except Exception as e:
            logging.error(f"Error generating snapshots: {e}")
            return []

    def generate_frame_snapshots(self, frame_annotations: List[Dict], results: List[Dict],
                               violations: List[Dict] = None, max_frames: int = 20) -> List[Dict]:
        """Generate individual frame snapshots with overlays for manual verification"""
        try:
            snapshot_paths = []
            frames_with_detections = []

            logging.info(f"Generating frame snapshots from {len(frame_annotations)} annotations and {len(results)} results")

            # Find frames that have plate detections
            for annotation in frame_annotations:
                frame_number = annotation.get('frame_number', 0)
                frame_image = annotation.get('frame_image')

                if frame_image is not None:
                    # Check if this frame has any plate detections
                    frame_plates = [r for r in results if r.get('frame_number') == frame_number]
                    if frame_plates:
                        frames_with_detections.append({
                            'frame_number': frame_number,
                            'frame_image': frame_image,
                            'annotation': annotation,
                            'plates': frame_plates
                        })
                        logging.debug(f"Frame {frame_number} has {len(frame_plates)} plates, will generate snapshot")

            logging.info(f"Found {len(frames_with_detections)} frames with detections")
            if frames_with_detections:
                logging.info(f"Sample frame with detection: frame {frames_with_detections[0]['frame_number']}")
            # Log frame_numbers from results
            result_frame_numbers = set(r.get('frame_number', 0) for r in results)
            logging.info(f"Result frame_numbers: {sorted(result_frame_numbers)}")
            # Log frame_numbers from frame_annotations
            annotation_frame_numbers = set(a.get('frame_number', 0) for a in frame_annotations)
            logging.info(f"Annotation frame_numbers: {sorted(annotation_frame_numbers)}")

            # Sort by frame number and limit to max_frames
            frames_with_detections.sort(key=lambda x: x['frame_number'])
            selected_frames = frames_with_detections[:max_frames]

            for i, frame_data in enumerate(selected_frames):
                frame_number = frame_data['frame_number']
                frame_image = frame_data['frame_image']
                annotation = frame_data['annotation']
                plates = frame_data['plates']

                # Add annotations to the frame before saving
                num_plates = len(plates)
                snapshot_name = f"frame_{frame_number:06d}_{num_plates}_plates.jpg"
                snapshot_path = os.path.join(self.snapshots_dir, snapshot_name)

                # Apply annotations to the frame
                # Extract overlays from the frame annotation
                overlays = annotation.get('overlays', [])
                annotated_frame = self._add_frame_annotations(frame_image, frame_number, violations or [], overlays)
                success = cv2.imwrite(snapshot_path, annotated_frame)
                if success:
                    snapshot_paths.append({
                        'frame_number': frame_number,
                        'filename': snapshot_name,
                        'plates': plates
                    })
                    logging.info(f"Generated annotated frame snapshot: {snapshot_path}")
                else:
                    logging.error(f"Failed to save frame snapshot: {snapshot_path}")

            logging.info(f"Generated {len(snapshot_paths)} frame snapshots for manual verification")
            return snapshot_paths

        except Exception as e:
            logging.error(f"Error generating frame snapshots: {e}")
            return []

    def _draw_overlay(self, frame: np.ndarray, overlay: Dict):
        """Draw a single overlay on the frame"""
        overlay_type = overlay.get('type', '')

        if overlay_type == 'text':
            cv2.putText(frame, overlay.get('text', ''),
                       overlay.get('position', (10, 50)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       overlay.get('font_scale', 0.7),
                       overlay.get('color', (255, 255, 255)),
                       overlay.get('thickness', 2))

        elif overlay_type == 'rectangle':
            cv2.rectangle(frame,
                         overlay.get('start_point', (0, 0)),
                         overlay.get('end_point', (100, 100)),
                         overlay.get('color', (0, 255, 0)),
                         overlay.get('thickness', 2))

        elif overlay_type == 'line':
            cv2.line(frame,
                    overlay.get('start_point', (0, 0)),
                    overlay.get('end_point', (100, 100)),
                    overlay.get('color', (0, 0, 255)),
                    overlay.get('thickness', 2))

        elif overlay_type == 'circle':
            cv2.circle(frame,
                      overlay.get('center', (50, 50)),
                      overlay.get('radius', 20),
                      overlay.get('color', (255, 255, 255)),
                      overlay.get('thickness', 2))

        elif overlay_type == 'filled_circle':
            cv2.circle(frame,
                      overlay.get('center', (50, 50)),
                      overlay.get('radius', 20),
                      overlay.get('color', (255, 255, 255)),
                      -1)  # Filled

    def generate_all_outputs(self, video_path: str, violations: List[Dict],
                           frame_annotations: List[Dict] = None,
                           metadata: Dict = None, results: List[Dict] = None) -> Dict[str, str]:
        """Generate all output files"""
        outputs = {}

        # Generate annotated video
        if video_path and os.path.exists(video_path):
            annotated_video = self.generate_annotated_video(video_path, violations, frame_annotations)
            if annotated_video:
                outputs['annotated_video'] = annotated_video

        # Generate reports
        csv_report = self.generate_csv_report(violations, metadata)
        if csv_report:
            outputs['csv_report'] = csv_report

        json_report = self.generate_json_report(violations, metadata)
        if json_report:
            outputs['json_report'] = json_report

        # Generate snapshots
        snapshots = self.generate_snapshots(violations)
        if snapshots:
            outputs['snapshots'] = snapshots

        # Generate frame snapshots for manual verification
        if frame_annotations and results:
            frame_snapshots = self.generate_frame_snapshots(frame_annotations, results, violations)
            if frame_snapshots:
                outputs['frame_snapshots'] = frame_snapshots

        logging.info(f"Generated {len(outputs)} types of outputs")
        return outputs

    def cleanup_temp_files(self):
        """Clean up temporary files if needed"""
        # Implementation for cleanup if necessary
        pass

    def get_output_summary(self) -> Dict:
        """Get summary of generated outputs"""
        return {
            'output_directory': self.output_dir,
            'annotated_video': self.annotated_video_path,
            'csv_report': self.report_csv_path,
            'json_report': self.report_json_path,
            'snapshots_directory': self.snapshots_dir
        }
