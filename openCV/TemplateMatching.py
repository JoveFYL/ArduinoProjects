import numpy as np
import cv2
import time
from collections import defaultdict
import math

class WholeObjectCounter:
    def __init__(self, template_path, detection_method='combined', belt_direction='horizontal'):
        self.detection_method = detection_method
        self.belt_direction = belt_direction
        
        # Load template image
        self.template_color = cv2.imread(template_path)
        self.template_gray = cv2.cvtColor(self.template_color, cv2.COLOR_BGR2GRAY)
        
        if self.template_gray is None:
            raise ValueError(f"Could not load template image: {template_path}")
        
        # Template matching parameters
        self.template_scales = [0.8, 0.9, 1.0, 1.1, 1.2]  # Multiple scales for robustness
        self.match_threshold = 0.6  # Minimum correlation for template matching
        
        # Contour detection parameters
        self.min_contour_area = 1000
        self.max_contour_area = 50000
        self.circularity_threshold = 0.7  # For circular objects like knobs
        
        # Tracking variables
        self.tracked_objects = {}
        self.next_object_id = 0
        self.frame_count = 0
        self.total_count = 0
        
        # Tracking parameters
        self.max_distance = 100
        self.max_frames_missing = 30
        self.min_travel_distance = 50
        
        # Zone setup
        self.entry_zone = None
        self.exit_zone = None
        self.tracking_zone = None
        
    def setup_zones(self, frame_shape):
        """Define detection zones"""
        height, width = frame_shape[:2]
        
        if self.belt_direction == 'horizontal':
            self.entry_zone = (0, 0, width//4, height)
            self.exit_zone = (3*width//4, 0, width//4, height)
            self.tracking_zone = (width//8, 0, 3*width//4, height)
        else:
            self.entry_zone = (0, 0, width, height//4)
            self.exit_zone = (0, 3*height//4, width, height//4)
            self.tracking_zone = (0, height//8, width, 3*height//4)
    
    def detect_objects_template_matching(self, frame):
        """Detect objects using multi-scale template matching"""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = []
        
        h, w = self.template_gray.shape
        
        for scale in self.template_scales:
            # Resize template
            scaled_template = cv2.resize(self.template_gray, 
                                       (int(w * scale), int(h * scale)))
            
            # Template matching
            result = cv2.matchTemplate(frame_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            
            # Find locations above threshold
            locations = np.where(result >= self.match_threshold)
            
            for pt in zip(*locations[::-1]):  # Switch x and y
                # Calculate center and confidence
                center_x = pt[0] + scaled_template.shape[1] // 2
                center_y = pt[1] + scaled_template.shape[0] // 2
                confidence = result[pt[1], pt[0]]
                
                # Check if this detection is too close to existing ones (non-max suppression)
                too_close = False
                for existing in detections:
                    distance = math.sqrt((center_x - existing['center'][0])**2 + 
                                       (center_y - existing['center'][1])**2)
                    if distance < min(scaled_template.shape) * 0.5:
                        if confidence > existing['confidence']:
                            # Replace with better detection
                            existing['center'] = (center_x, center_y)
                            existing['confidence'] = confidence
                            existing['size'] = (scaled_template.shape[1], scaled_template.shape[0])
                        too_close = True
                        break
                
                if not too_close:
                    detections.append({
                        'center': (center_x, center_y),
                        'confidence': confidence,
                        'size': (scaled_template.shape[1], scaled_template.shape[0]),
                        'scale': scale
                    })
        
        return detections
    
    def detect_objects_contour_based(self, frame):
        """Detect circular objects using contour analysis"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 2)
        
        # Use adaptive threshold for better edge detection
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if self.min_contour_area < area < self.max_contour_area:
                # Calculate circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * math.pi * area / (perimeter * perimeter)
                    
                    # Filter by circularity (for circular objects like knobs)
                    if circularity > self.circularity_threshold:
                        # Get bounding rectangle
                        x, y, w, h = cv2.boundingRect(contour)
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Calculate confidence based on circularity and area
                        confidence = circularity * min(1.0, area / self.max_contour_area)
                        
                        detections.append({
                            'center': (center_x, center_y),
                            'confidence': confidence,
                            'size': (w, h),
                            'area': area,
                            'circularity': circularity,
                            'contour': contour
                        })
        
        return detections
    
    def detect_objects_hough_circles(self, frame):
        """Detect circular objects using Hough Circle Transform"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Hough Circle detection
        circles = cv2.HoughCircles(blurred, 
                                 cv2.HOUGH_GRADIENT, 
                                 dp=1, 
                                 minDist=50,  # Minimum distance between circle centers
                                 param1=50,   # Upper threshold for edge detection
                                 param2=30,   # Accumulator threshold for center detection
                                 minRadius=20, # Minimum circle radius
                                 maxRadius=100) # Maximum circle radius
        
        detections = []
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Calculate confidence based on circle quality
                confidence = min(1.0, r / 50.0)  # Normalize radius to confidence
                
                detections.append({
                    'center': (x, y),
                    'confidence': confidence,
                    'size': (r*2, r*2),
                    'radius': r
                })
        
        return detections
    
    def detect_objects_combined(self, frame):
        """Combine multiple detection methods for better accuracy"""
        detections = []
        
        # Combine all methods
        template_dets = self.detect_objects_template_matching(frame)
        contour_dets = self.detect_objects_contour_based(frame)
        hough_dets = self.detect_objects_hough_circles(frame)
        
        # Merge detections with confidence boosting for overlapping detections
        all_detections = template_dets + contour_dets + hough_dets
        detections = self.merge_overlapping_detections(all_detections)
        
        return detections
    
    def merge_overlapping_detections(self, all_detections, overlap_threshold=50):
        """Merge overlapping detections from different methods"""
        if not all_detections:
            return []
        
        merged = []
        used = [False] * len(all_detections)
        
        for i, det1 in enumerate(all_detections):
            if used[i]:
                continue
                
            # Start new merged detection
            merged_det = det1.copy()
            overlapping = [det1]
            used[i] = True
            
            # Find overlapping detections
            for j, det2 in enumerate(all_detections):
                if used[j] or i == j:
                    continue
                    
                distance = math.sqrt((det1['center'][0] - det2['center'][0])**2 + 
                                   (det1['center'][1] - det2['center'][1])**2)
                
                if distance < overlap_threshold:
                    overlapping.append(det2)
                    used[j] = True
            
            # If multiple detections overlap, boost confidence and average position
            if len(overlapping) > 1:
                total_confidence = sum(det['confidence'] for det in overlapping)
                avg_x = sum(det['center'][0] for det in overlapping) / len(overlapping)
                avg_y = sum(det['center'][1] for det in overlapping) / len(overlapping)
                
                merged_det['center'] = (int(avg_x), int(avg_y))
                merged_det['confidence'] = min(1.0, total_confidence * 1.2)  # Boost confidence
                merged_det['method_count'] = len(overlapping)
            
            merged.append(merged_det)
        
        return merged
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def point_in_zone(self, point, zone):
        """Check if point is within a zone"""
        x, y = point
        zx, zy, zw, zh = zone
        return zx <= x <= zx + zw and zy <= y <= zy + zh
    
    def update_tracking(self, detections, frame_shape=None):
        """Update object tracking and counting"""
        self.frame_count += 1
        
        # Initialize zones if not already done
        if self.entry_zone is None and frame_shape is not None:
            self.setup_zones(frame_shape)
        
        # Match detections to existing objects
        unmatched_detections = list(range(len(detections)))
        
        # Update existing objects
        for obj_id, obj_data in list(self.tracked_objects.items()):
            if self.frame_count - obj_data['last_seen'] > self.max_frames_missing:
                del self.tracked_objects[obj_id]
                continue
            
            # Find closest detection
            last_pos = obj_data['positions'][-1]
            best_match_idx = None
            min_distance = float('inf')
            
            for det_idx in unmatched_detections:
                distance = self.calculate_distance(last_pos, detections[det_idx]['center'])
                if distance < self.max_distance and distance < min_distance:
                    min_distance = distance
                    best_match_idx = det_idx
            
            if best_match_idx is not None:
                # Update object
                obj_data['positions'].append(detections[best_match_idx]['center'])
                obj_data['last_seen'] = self.frame_count
                obj_data['confidence'] = max(obj_data['confidence'], detections[best_match_idx]['confidence'])
                unmatched_detections.remove(best_match_idx)
                
                # Check if should be counted
                if not obj_data['counted'] and len(obj_data['positions']) > 8:
                    if self.should_count_object(obj_data):
                        obj_data['counted'] = True
                        self.total_count += 1
                        print(f"WHOLE OBJECT COUNTED! Total: {self.total_count}")
        
        # Create new objects for unmatched detections
        for det_idx in unmatched_detections:
            detection = detections[det_idx]
            
            # Only track high-confidence detections
            if detection['confidence'] > 0.5:
                # Check zones only if they're initialized
                in_valid_zone = True
                if self.entry_zone is not None and self.tracking_zone is not None:
                    in_valid_zone = (self.point_in_zone(detection['center'], self.entry_zone) or 
                                   self.point_in_zone(detection['center'], self.tracking_zone))
                
                if in_valid_zone:
                    self.tracked_objects[self.next_object_id] = {
                        'positions': [detection['center']],
                        'last_seen': self.frame_count,
                        'counted': False,
                        'confidence': detection['confidence'],
                        'birth_frame': self.frame_count
                    }
                    print(f"New object {self.next_object_id} detected")
                    self.next_object_id += 1
    
    def should_count_object(self, obj_data):
        """Determine if object should be counted"""
        positions = obj_data['positions']
        
        if len(positions) < 5:
            return False
        
        # Calculate travel distance
        total_distance = 0
        for i in range(1, len(positions)):
            total_distance += self.calculate_distance(positions[i-1], positions[i])
        
        if total_distance < self.min_travel_distance:
            return False
        
        # Check movement direction
        start_pos = positions[0]
        current_pos = positions[-1]
        
        if self.belt_direction == 'horizontal':
            movement = current_pos[0] - start_pos[0]
            return movement > self.min_travel_distance * 0.6
        else:
            movement = current_pos[1] - start_pos[1]
            return movement > self.min_travel_distance * 0.6
    
    def draw_visualization(self, frame, detections):
        """Draw detection and tracking visualization"""
        if self.entry_zone is None:
            self.setup_zones(frame.shape)
        
        # Draw zones
        cv2.rectangle(frame, (self.entry_zone[0], self.entry_zone[1]), 
                     (self.entry_zone[0] + self.entry_zone[2], self.entry_zone[1] + self.entry_zone[3]), 
                     (0, 255, 0), 2)
        cv2.rectangle(frame, (self.exit_zone[0], self.exit_zone[1]), 
                     (self.exit_zone[0] + self.exit_zone[2], self.exit_zone[1] + self.exit_zone[3]), 
                     (0, 0, 255), 2)
        
        # Draw current detections
        for detection in detections:
            center = detection['center']
            confidence = detection['confidence']
            
            # Draw detection circle
            radius = detection.get('radius', 30)
            cv2.circle(frame, (int(center[0]), int(center[1])), radius, (255, 0, 0), 2)
            
            # Draw confidence
            cv2.putText(frame, f"{confidence:.2f}", 
                       (int(center[0]), int(center[1]-radius-10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Draw method info if available
            # if 'method_count' in detection:
            #     cv2.putText(frame, f"Methods:{detection['method_count']}", 
            #                (int(center[0]), int(center[1]+radius+20)), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Draw tracked objects
        for obj_id, obj_data in self.tracked_objects.items():
            positions = obj_data['positions']
            
            # # Draw trajectory
            # if len(positions) > 1:
            #     for i in range(1, len(positions)):
            #         cv2.line(frame, (int(positions[i-1][0]), int(positions[i-1][1])), 
            #                 (int(positions[i][0]), int(positions[i][1])), (0, 255, 255), 2)
            
            # Draw object ID and status
            if positions:
                pos = positions[-1]
                color = (0, 255, 0) if obj_data['counted'] else (255, 255, 255)
                cv2.putText(frame, f"ID:{obj_id}", (int(pos[0]), int(pos[1]-40)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw info
        cv2.putText(frame, f'Method: {self.detection_method}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Total Count: {self.total_count}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f'Detections: {len(detections)}', 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'Tracked: {len(self.tracked_objects)}', 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def process_video(self, video_path):
        """Process video with whole object detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print("Error opening video")
            return
        
        paused = False
        frame_count = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
            
            # Speed up video by processing only every 10 frames
            frame_count += 1
            if frame_count % 10 != 0:
              continue
            
            frame = frame[400:700, 300:1200]
            # frame = frame[:300, 1200:2000]

            # Detect objects
            detections = self.detect_objects_combined(frame)
            
            # Update tracking (pass frame shape for zone initialization)
            self.update_tracking(detections, frame.shape)
            
            # Draw visualization
            vis_frame = self.draw_visualization(frame, detections)
            
            cv2.imshow('Whole Object Counter', vis_frame)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.total_count = 0
                self.tracked_objects.clear()
                self.next_object_id = 0
                print("Count reset!")
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nFinal Count: {self.total_count}")
        return self.total_count

# Usage examples for different detection methods:
if __name__ == "__main__":
    # For circular knobs like yours - try Hough circles first
    counter = WholeObjectCounter("./blackKnob.jpg", 
                               detection_method='combined',
                               belt_direction='horizontal')
    
    # Or try template matching for exact shape matching
    # counter = WholeObjectCounter("./knob_image.jpg", 
    #                            detection_method='template_matching',
    #                            belt_direction='horizontal')
    
    # Or use combined method for best results
    # counter = WholeObjectCounter("./knob_image.jpg", 
    #                            detection_method='combined',
    #                            belt_direction='horizontal')
    
    final_count = counter.process_video("./knobWeird.MOV")
    print(f"Total objects counted: {final_count}")
