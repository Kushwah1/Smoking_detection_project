import cv2
import numpy as np
import os
import time
import datetime
from PIL import Image, ImageEnhance
import threading

# Constants
IMAGE_SIZE = (224, 224)
CLASSES = ["not_smoking", "smoking"]

# Global configuration settings
AUTO_SAVE = True                 # Automatically save smoking detection events
AUTO_SAVE_INTERVAL = 3           # Minimum seconds between auto-saves
AUTO_SAVE_CONFIDENCE = 0.60      # Confidence threshold for auto-save
AUTO_SAVE_DURATION = 2           # How many seconds to keep saving frames after detection

# Global variable for temporal smoothing
last_results = []

def preprocess_frame(frame):
    """Preprocess a frame for detection"""
    try:
        # Resize frame to match model input size
        resized = cv2.resize(frame, IMAGE_SIZE)
        # Convert to RGB (OpenCV uses BGR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        normalized = rgb / 255.0
        return normalized
    except Exception as e:
        print(f"Error preprocessing frame: {e}")
        return None

def enhance_frame(frame):
    """Enhance the frame for better detection"""
    try:
        # Convert to PIL image for enhancement
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Apply sharpening filter
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(1.6)  # Adjusted sharpening
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.4)  # Adjusted contrast
        
        # Enhance color
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(1.25)  # Adjusted color
        
        # Convert back to OpenCV format
        enhanced_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return enhanced_frame
    except Exception as e:
        print(f"Error enhancing frame: {e}")
        return frame  # Return original frame if enhancement fails

def detect_smoking(frame):
    """Detect smoking in a frame using multiple techniques for stability"""
    # Clone the frame to avoid modifying the original
    processed_frame = frame.copy()
    
    # Result storage
    detection_results = []
    
    try:
        # Get frame dimensions for percentage calculations
        height, width = processed_frame.shape[:2]
        total_pixels = height * width
        
        # First check if this is a portrait photo (face detection)
        # This is important to reduce false positives in portrait photos like the Rohit Sharma image
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        is_portrait = False
        portrait_face_size = 0
        face_detected = False
        faces = []
        
        try:
            # Load the face cascade - use absolute path to avoid issues
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            # Print the path for debugging
            print(f"Using face cascade: {cascade_path}")
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Verify cascade loaded properly
            if face_cascade.empty():
                print("WARNING: Face cascade failed to load - using fallback detection")
            else:
                # Run face detection with reasonable parameters
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                # Check if faces were detected
                if len(faces) > 0:
                    face_detected = True
                    print(f"Face detection: {len(faces)} faces found")
                    
                    # Check if this is a portrait photo (large face relative to image)
                    for (fx, fy, fw, fh) in faces:
                        # Calculate face size relative to image
                        face_area = fw * fh
                        face_ratio = face_area / total_pixels
                        
                        # Draw face rectangles (in blue)
                        cv2.rectangle(processed_frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
                        
                        print(f"Face ratio: {face_ratio:.4f} (ratio > 0.1 indicates a portrait)")
                        
                        # If face occupies significant portion of image, it's likely a portrait
                        if face_ratio > 0.08:  # Slightly lower threshold for better detection
                            is_portrait = True
                            portrait_face_size = face_area
                            # Draw text indicating portrait detection
                            cv2.putText(processed_frame, "PORTRAIT", (fx, fy-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                            
                            print("PORTRAIT MODE ACTIVATED - Using stricter detection thresholds")
        except Exception as e:
            print(f"Error in face detection: {e}")
            # Continue without face detection
        
        # 1. Color-based detection for cigarette glow
        hsv = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2HSV)
        
        # Orange/red color range for cigarette glow
        lower_orange = np.array([5, 150, 200])
        upper_orange = np.array([25, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((3, 3), np.uint8)
        mask_orange = cv2.morphologyEx(mask_orange, cv2.MORPH_OPEN, kernel)
        
        # Count orange/red pixels (potential cigarette glow)
        orange_pixel_count = cv2.countNonZero(mask_orange)
        orange_ratio = orange_pixel_count / total_pixels
        
        # For portrait images, require much higher orange pixel count
        orange_threshold = 0.01 if is_portrait else 0.005
        min_orange_pixels = 200 if is_portrait else 50
        
        if orange_ratio > orange_threshold and orange_pixel_count > min_orange_pixels:
            # Analyze the shape of the orange areas to check if they're cigarette-like
            contours_orange, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_orange_contours = 0
            
            for contour in contours_orange:
                area = cv2.contourArea(contour)
                if 20 < area < 800:
                    valid_orange_contours += 1
            
            if valid_orange_contours > 0:
                detection_results.append(("smoking", min(orange_ratio * 70 + 0.1, 0.7)))
        
        # 2. Shape-based detection for cigarette
        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection with thresholds adjusted for portraits
        edge_thresh1 = 70 if is_portrait else 50
        edge_thresh2 = 180 if is_portrait else 150
        edges = cv2.Canny(blurred, edge_thresh1, edge_thresh2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours for potential cigarette shapes
        cigarette_contours = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h != 0 else 0
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate shape complexity
            complexity = perimeter**2 / (4 * np.pi * area) if area > 0 else float('inf')
            
            # For portraits, use stricter criteria
            if is_portrait:
                # Stricter filtering for portraits to prevent false positives
                if (4.5 < aspect_ratio < 8.0 and 
                    150 < area < 2000 and 
                    complexity < 3.0):
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    
                    if solidity > 0.8:
                        cigarette_contours.append(contour)
            else:
                # Regular filtering for non-portraits
                if (4.0 < aspect_ratio < 10.0 and 
                    100 < area < 5000 and 
                    complexity < 4.0):
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area > 0 else 0
                    
                    if solidity > 0.7:
                        cigarette_contours.append(contour)
        
        # Draw cigarette contours for visualization (in red)
        cv2.drawContours(processed_frame, cigarette_contours, -1, (0, 0, 255), 2)
        
        # For portraits, require more cigarette contours
        min_contours = 2 if is_portrait else 1
        
        if len(cigarette_contours) >= min_contours:
            confidence = min(0.4 + (len(cigarette_contours) * 0.1), 0.7)
            detection_results.append(("smoking", confidence))
        
        # 3. White/gray color detection for cigarette body
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 35, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Apply morphological operations to clean up
        mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        
        # Create mask for analyzing white regions in cigarette-like shapes
        if cigarette_contours:
            white_cigarette_mask = np.zeros_like(mask_white)
            for contour in cigarette_contours:
                cv2.drawContours(white_cigarette_mask, [contour], -1, 255, -1)
            
            # Apply white mask to potential cigarette contours
            white_in_cigarette = cv2.bitwise_and(mask_white, white_cigarette_mask)
            white_cig_pixel_count = cv2.countNonZero(white_in_cigarette)
            
            # More strict for portraits
            min_white_pixels = 100 if is_portrait else 50
            
            if white_cig_pixel_count > min_white_pixels:
                detection_results.append(("smoking", 0.65))
        
        # 4. Hand-to-mouth gesture detection
        try:
            # Use skin detection to find potential hand regions
            lower_skin = np.array([0, 15, 60], dtype=np.uint8)
            upper_skin = np.array([30, 170, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Apply morphological operations to clean up
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours of skin regions
            skin_contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter for hand-sized contours
            hand_contours = []
            for contour in skin_contours:
                area = cv2.contourArea(contour)
                # For portraits, use more specific hand size criteria
                if is_portrait:
                    # In portraits, hands should be smaller than faces
                    if 500 < area < portrait_face_size * 0.8:
                        hand_contours.append(contour)
                else:
                    if 500 < area < 20000:
                        hand_contours.append(contour)
            
            # Draw potential hand regions (in green)
            cv2.drawContours(processed_frame, hand_contours, -1, (0, 255, 0), 1)
            
            # Check if any cigarette contour is near a hand contour
            close_to_hand = False
            if cigarette_contours and hand_contours:
                for cig_contour in cigarette_contours:
                    cig_x, cig_y, cig_w, cig_h = cv2.boundingRect(cig_contour)
                    cig_center = (cig_x + cig_w//2, cig_y + cig_h//2)
                    
                    for hand_contour in hand_contours:
                        hand_x, hand_y, hand_w, hand_h = cv2.boundingRect(hand_contour)
                        # Check if cigarette is inside or near hand
                        if (hand_x - cig_w <= cig_center[0] <= hand_x + hand_w + cig_w and
                            hand_y - cig_h <= cig_center[1] <= hand_y + hand_h + cig_h):
                            close_to_hand = True
                            # Draw a connection line to show the detection (in yellow)
                            cv2.line(processed_frame, cig_center, 
                                    (hand_x + hand_w//2, hand_y + hand_h//2), 
                                    (0, 255, 255), 2)
                            break
                    
                    if close_to_hand:
                        break
                
                if close_to_hand:
                    # Higher confidence if cigarette is near hand
                    detection_results.append(("smoking", 0.75))
            
            # 5. Face-cigarette proximity check
            try:
                if faces.size > 0 and cigarette_contours:
                    # Check if any cigarette is near the mouth/face
                    cigarette_near_mouth = False
                    
                    for (fx, fy, fw, fh) in faces:
                        # Define mouth region (lower 1/3 of face)
                        mouth_y = fy + int(fh * 2/3)
                        mouth_h = fh - int(fh * 2/3)
                        
                        # For portraits, we need to be more precise
                        proximity_x = 15 if is_portrait else 20
                        proximity_y = 10 if is_portrait else 15
                        
                        # Define face region
                        face_expanded = (fx-proximity_x, fy-proximity_y, fw+proximity_x*2, fh+proximity_y*2)
                        
                        for contour in cigarette_contours:
                            x, y, w, h = cv2.boundingRect(contour)
                            cig_center = (x + w//2, y + h//2)
                            
                            # For portraits, cigarette should specifically be near the mouth
                            if is_portrait:
                                if (fx-5 < cig_center[0] < fx+fw+5 and 
                                    mouth_y-5 < cig_center[1] < mouth_y+mouth_h+10):
                                    cigarette_near_mouth = True
                                    confidence = 0.8
                                    # Draw a highlight around this cigarette
                                    cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                                    break
                            else:
                                # For non-portraits, being near face is enough
                                if ((face_expanded[0] < cig_center[0] < face_expanded[0] + face_expanded[2]) and
                                    (face_expanded[1] < cig_center[1] < face_expanded[1] + face_expanded[3])):
                                    # Higher confidence if near mouth specifically
                                    if (fx < cig_center[0] < fx+fw and 
                                        mouth_y-10 < cig_center[1] < mouth_y+mouth_h+20):
                                        cigarette_near_mouth = True
                                        confidence = 0.8
                                    else:
                                        cigarette_near_mouth = True
                                        confidence = 0.7
                                    
                                    # Draw a highlight around this cigarette
                                    cv2.rectangle(processed_frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                                    break
                    
                    # Add confidence if cigarette is near a face/mouth
                    if cigarette_near_mouth:
                        detection_results.append(("smoking", confidence))
            except:
                pass
                
        except Exception as e:
            print(f"Error in contextual detection: {e}")
        
        # CRITICAL: For portrait images, require multiple indicators
        if is_portrait:
            print(f"Portrait mode: Found {len(detection_results)} potential smoking indicators")
            # If this is a portrait, we need at least 2 strong indicators of smoking
            if len(detection_results) < 2:
                print("Portrait mode: Not enough indicators for positive detection")
                # Clear results if not enough indicators for a portrait
                detection_results = []
                # Add text to indicate portrait requires higher confidence
                cv2.putText(processed_frame, "PORTRAIT MODE: Higher threshold applied", 
                          (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # If we detected a face but didn't classify as portrait, be more cautious
        elif face_detected:
            print(f"Face detected but not portrait: Found {len(detection_results)} potential smoking indicators")
            # For images with faces that aren't portraits, require more evidence
            if len(detection_results) < 2:
                # Not enough indicators for a face image
                detection_results = []
    
    except Exception as e:
        print(f"Error in detection process: {e}")
        return "not_smoking", 0.75, processed_frame
    
    # Combine results
    if not detection_results:
        return "not_smoking", 0.8, processed_frame
    
    # Get the highest confidence smoking detection
    max_confidence = 0
    for result in detection_results:
        if result[0] == "smoking" and result[1] > max_confidence:
            max_confidence = result[1]
    
    # Apply temporal smoothing
    global last_results
    
    # For portraits, use higher threshold
    detection_threshold = 0.70 if is_portrait else 0.60
    
    last_results.append(("smoking" if max_confidence > detection_threshold else "not_smoking", max_confidence))
    if len(last_results) > 10:
        last_results.pop(0)
    
    # Count smoking vs not_smoking labels in last results
    smoking_count = sum(1 for r in last_results if r[0] == "smoking")
    smoking_ratio = smoking_count / len(last_results)
    
    # Use stricter smoothing for portraits
    smoothing_threshold = 0.7 if is_portrait else 0.6
    
    if smoking_ratio > smoothing_threshold or (len(last_results) < 3 and max_confidence > 0.75):
        avg_confidence = sum(r[1] for r in last_results if r[0] == "smoking") / max(smoking_count, 1)
        return "smoking", avg_confidence, processed_frame
    else:
        return "not_smoking", 0.8, processed_frame

def save_detection_frame(frame, label, confidence, output_dir):
    """Save a detection frame with timestamp"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"detection_{label}_{confidence:.2f}_{timestamp}.jpg")
    cv2.imwrite(output_path, frame)
    return output_path

def run_detection():
    """Run real-time smoking detection"""
    print("Starting real-time smoking detector...")
    
    # Global variables
    global AUTO_SAVE, AUTO_SAVE_CONFIDENCE
    
    try:
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam!")
            return
        
        print("Webcam opened successfully!")
        print("Press 'q' to quit, 's' to save a snapshot")
        if AUTO_SAVE:
            print(f"Auto-save is ENABLED - will save images when smoking is detected (confidence > {AUTO_SAVE_CONFIDENCE:.2f})")
        
        # Get webcam properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Webcam properties: {width}x{height} at {fps} FPS")
        
        # Create output directory
        output_dir = 'detection_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        # Create a subdirectory for auto-saved images
        auto_save_dir = os.path.join(output_dir, 'auto_saved')
        if not os.path.exists(auto_save_dir):
            os.makedirs(auto_save_dir)
            print(f"Created auto-save directory: {auto_save_dir}")
        
        # Initialize FPS calculation
        frame_count = 0
        start_time = time.time()
        fps_display = 0
        
        # Auto-save tracking
        last_auto_save_time = 0
        auto_save_active_until = 0
        
        # Create color map for different classes
        color_map = {
            "not_smoking": (0, 255, 0),  # Green
            "smoking": (0, 0, 255),  # Red
        }
        
        # Initialize a separate processing thread
        processing_active = False
        processing_result = None
        
        def process_frame_thread(frame):
            nonlocal processing_result, processing_active
            # Enhance the frame
            enhanced_frame = enhance_frame(frame)
            # Detect smoking
            label, confidence, processed_frame = detect_smoking(enhanced_frame)
            processing_result = (label, confidence, processed_frame)
            processing_active = False
        
        # Main loop
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            # Check if frame was captured
            if not ret:
                print("Error: Failed to capture frame! Trying to reconnect...")
                # Attempt to reconnect
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Reconnection failed. Exiting.")
                    break
                continue
            
            # Make a copy for display
            display_frame = frame.copy()
            
            # Start processing in a separate thread if not already active
            if not processing_active and frame_count % 3 == 0:  # Process every 3rd frame
                processing_active = True
                processing_thread = threading.Thread(target=process_frame_thread, args=(frame,))
                processing_thread.daemon = True
                processing_thread.start()
            
            # Update display if processing result is available
            if processing_result is not None:
                label, confidence, processed_display = processing_result
                
                # Get color based on label
                color = color_map.get(label, (0, 255, 0))
                
                # Calculate FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    fps_display = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Draw results on frame
                cv2.putText(display_frame, f"{label.upper()}: {confidence:.2f}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Add FPS info
                cv2.putText(display_frame, f"FPS: {fps_display:.1f}", 
                          (width - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add auto-save status
                if AUTO_SAVE:
                    if time.time() < auto_save_active_until:
                        cv2.putText(display_frame, "AUTO-SAVING ACTIVE", 
                                  (width // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                # Add instructions
                cv2.putText(display_frame, "Press 'q' to quit, 's' to save snapshot", 
                          (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(display_frame, timestamp, 
                          (width - 200, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Combine with processed display (showing contours)
                alpha = 0.3  # Transparency factor
                overlay = cv2.addWeighted(display_frame, 1 - alpha, processed_display, alpha, 0)
                display_frame = overlay
                
                # Auto-save logic
                current_time = time.time()
                time_since_last_save = current_time - last_auto_save_time
                
                # Check if we need to start auto-saving
                if (AUTO_SAVE and label == "smoking" and confidence >= AUTO_SAVE_CONFIDENCE and 
                    time_since_last_save >= AUTO_SAVE_INTERVAL):
                    # Start auto-save period
                    auto_save_active_until = current_time + AUTO_SAVE_DURATION
                    print(f"⚠️ SMOKING DETECTED! Auto-saving for {AUTO_SAVE_DURATION} seconds...")
                
                # Save frame if auto-save is active
                if AUTO_SAVE and current_time < auto_save_active_until:
                    # Save frame with detection info
                    save_path = os.path.join(auto_save_dir, f"smoking_{confidence:.2f}_{int(current_time)}.jpg")
                    cv2.imwrite(save_path, display_frame)
                    last_auto_save_time = current_time
                    
                    # Add a red border to indicate recording
                    cv2.rectangle(display_frame, (0, 0), (width-1, height-1), (0, 0, 255), 3)
            
            # Display the frame
            cv2.imshow('Smoking Detection', display_frame)
            
            # Process key presses
            key = cv2.waitKey(1) & 0xFF
            
            # 'q' to quit
            if key == ord('q'):
                print("Quitting...")
                break
            
            # 's' to save snapshot
            elif key == ord('s') and processing_result is not None:
                label, confidence, _ = processing_result
                saved_path = save_detection_frame(display_frame, label, confidence, output_dir)
                print(f"Snapshot saved as {saved_path}")
            
            # 'a' to toggle auto-save
            elif key == ord('a'):
                AUTO_SAVE = not AUTO_SAVE
                print(f"Auto-save {'ENABLED' if AUTO_SAVE else 'DISABLED'}")
            
            # '+' to increase auto-save confidence threshold
            elif key == ord('+') or key == ord('='):
                AUTO_SAVE_CONFIDENCE = min(AUTO_SAVE_CONFIDENCE + 0.05, 0.95)
                print(f"Auto-save confidence threshold increased to {AUTO_SAVE_CONFIDENCE:.2f}")
            
            # '-' to decrease auto-save confidence threshold
            elif key == ord('-'):
                AUTO_SAVE_CONFIDENCE = max(AUTO_SAVE_CONFIDENCE - 0.05, 0.5)
                print(f"Auto-save confidence threshold decreased to {AUTO_SAVE_CONFIDENCE:.2f}")
        
        # Release resources
        print("Releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        print("Detection complete!")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        # Cleanup if exception occurs
        try:
            cap.release()
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    run_detection() 