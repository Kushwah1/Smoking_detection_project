import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import argparse
from datetime import datetime
from PIL import Image
import io

IMAGE_SIZE = (224, 224) # Add IMAGE_SIZE constant
CLASSES = ["none", "smoking", "alcohol", "both"]

def preprocess_frame_tf(image_path):
    """Preprocess an image file for model prediction with robust error handling"""
    try:
        # Special handling for WEBP files
        if image_path.lower().endswith('.webp'):
            print("WEBP file detected, using PIL for conversion...")
            try:
                from PIL import Image
                import numpy as np
                
                # Open and convert WEBP to RGB using PIL
                pil_image = Image.open(image_path).convert('RGB')
                
                # Save as temporary JPG for compatibility
                temp_jpg_path = image_path.replace('.webp', '_temp.jpg')
                pil_image.save(temp_jpg_path, 'JPEG')
                print(f"Converted WEBP to temporary JPEG: {temp_jpg_path}")
                
                # Use the temporary JPG path
                image_path = temp_jpg_path
            except Exception as e:
                print(f"WEBP conversion failed: {e}")
        
        # Standard processing path
        try:
            # First try using PIL for all image formats (most reliable)
            from PIL import Image
            import io
            import numpy as np
            
            # Open image with PIL
            print(f"Loading image using PIL: {image_path}")
            pil_image = Image.open(image_path).convert('RGB')
            img_array = np.array(pil_image)
            
            # Convert numpy array to tensor
            img = tf.convert_to_tensor(img_array, dtype=tf.float32)
            
            # Resize
            img = tf.image.resize(img, IMAGE_SIZE)
            
            # Normalize
            img = img / 255.0
            
            # Print image shape and range for debugging
            print(f"Image shape after preprocessing: {img.shape}")
            print(f"Image value range: [{tf.reduce_min(img).numpy()}, {tf.reduce_max(img).numpy()}]")
            
        except Exception as e:
            print(f"PIL loading failed, trying TensorFlow decoders: {e}")
            
            # Load the raw data from the file as a string
            img = tf.io.read_file(image_path)
            
            # Try different TensorFlow decoders
            try:
                # Try JPEG first
                img = tf.image.decode_jpeg(img, channels=3)
            except:
                try:
                    # Try PNG next
                    img = tf.image.decode_png(img, channels=3)
                except:
                    try:
                        # Last try with generic decoder
                        img = tf.image.decode_image(img, channels=3, expand_animations=False)
                    except Exception as e:
                        raise ValueError(f"Failed to decode image {image_path}: {e}")
            
            # Resize and normalize
            img = tf.image.resize(img, IMAGE_SIZE)
            img = tf.cast(img, tf.float32) / 255.0
        
        # Add batch dimension
        return tf.expand_dims(img, axis=0)
    
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        raise

def adapt_legacy_model_output(model, prediction, threshold=0.5):
    """
    Adapt the output of a legacy smoking detection model (binary) to our multi-class format
    This is a temporary solution until we train the proper multi-class model
    """
    # Convert binary output to multi-class format
    # Smoking detection model output: 0=not smoking, 1=smoking
    smoking_prob = float(prediction[0][0])
    
    # Check if it's smoking or not
    if smoking_prob > threshold:
        # It's smoking, so set smoking class to highest
        class_probs = [0.05, smoking_prob, 0.03, 0.02]  # [none, smoking, alcohol, both]
    else:
        # It's not smoking, so set none class to highest
        class_probs = [1 - smoking_prob, smoking_prob, 0.01, 0.01]  # [none, smoking, alcohol, both]
    
    return np.array(class_probs)

def detect_behavior(model, frame_path, is_legacy_model=False, threshold=0.5):
    # Print message first
    print("\n---------------------------------------")
    print(f"ANALYZING IMAGE: {frame_path}")
    print("---------------------------------------\n")
    
    # Preprocess the frame using TensorFlow
    processed_frame = preprocess_frame_tf(frame_path)
    
    # Make prediction
    raw_prediction = model.predict(processed_frame, verbose=0)
    
    # Handle legacy smoking detection model
    if is_legacy_model:
        print("\nUsing legacy smoking detection model - adapting output to multi-class format")
        predictions = adapt_legacy_model_output(model, raw_prediction, threshold)
    else:
        predictions = raw_prediction[0]
    
    # Get class with highest confidence
    predicted_class_idx = np.argmax(predictions)
    confidence = predictions[predicted_class_idx]
    label = CLASSES[predicted_class_idx]
    
    # Print raw predictions for all classes
    print("\n----- Raw Predictions -----")
    for i, class_name in enumerate(CLASSES):
        print(f"{class_name}: {predictions[i]:.4f}")
    
    # Print result with more visibility
    print("\n==================================")
    print(f"RESULT: {label.upper()} (Confidence: {confidence:.2f})")
    print(f"Raw prediction value: {confidence:.4f}")
    print("==================================\n")
    
    return label, confidence, predictions

def process_image(model, image_path, save_output=False, headless=False, is_legacy_model=False, threshold=0.5):
    # Read image using OpenCV for display purposes only
    display_frame = cv2.imread(image_path)
    if display_frame is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Detect behavior using the original image path for TF preprocessing
    label, confidence, predictions = detect_behavior(model, image_path, is_legacy_model, threshold)
    
    # Print result to console
    print(f"Prediction: {label} (Confidence: {confidence:.2f})")
    
    # If in headless mode, skip display and just return
    if headless:
        if save_output:
            output_dir = 'test_results'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"result_{timestamp}.jpg")
            
            # Set color based on label
            color_map = {
                "none": (0, 255, 0),  # Green
                "smoking": (0, 0, 255),  # Red
                "alcohol": (255, 0, 0),  # Blue
                "both": (255, 0, 255)  # Magenta
            }
            color = color_map.get(label, (0, 255, 0))
            
            # Draw results on the frame
            cv2.putText(display_frame, f"{label}: {confidence:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Add probabilities for all classes
            y_offset = 60
            for i, class_name in enumerate(CLASSES):
                prob_text = f"{class_name}: {predictions[i]:.4f}"
                cv2.putText(display_frame, prob_text, 
                      (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                y_offset += 25
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, timestamp, 
                   (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imwrite(output_path, display_frame)
            print(f"Result saved as: {output_path}")
        return
    
    # Set color based on label
    color_map = {
        "none": (0, 255, 0),  # Green
        "smoking": (0, 0, 255),  # Red
        "alcohol": (255, 0, 0),  # Blue
        "both": (255, 0, 255)  # Magenta
    }
    color = color_map.get(label, (0, 255, 0))
    
    # Draw results on the frame read by OpenCV
    cv2.putText(display_frame, f"{label}: {confidence:.2f}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    # Add probabilities for all classes
    y_offset = 60
    for i, class_name in enumerate(CLASSES):
        prob_text = f"{class_name}: {predictions[i]:.4f}"
        cv2.putText(display_frame, prob_text, 
              (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(display_frame, timestamp, 
               (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display image
    cv2.imshow('Behavior Detection', display_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save the output if requested
    if save_output:
        output_dir = 'test_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"result_{timestamp}.jpg")
        cv2.imwrite(output_path, display_frame)
        print(f"Result saved as: {output_path}")

def process_video(model, video_path, save_output=False, headless=False, is_legacy_model=False, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = None
    if save_output:
        output_dir = 'test_results'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(output_dir, f"result_{timestamp}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        print(f"Output will be saved as: {output_filename}")

    frame_count = 0
    temp_frame_path = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            display_frame = frame.copy()
            
            # Process every 5th frame to improve performance
            if frame_count % 5 != 0:
                if out:
                    out.write(display_frame)
                if not headless:
                    cv2.imshow('Behavior Detection', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue
            
            # Save frame temporarily to pass its path to detect_behavior
            temp_dir = 'temp_video_frames'
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            temp_frame_path = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(temp_frame_path, frame)

            # Detect behavior using the temporary frame path
            label, confidence, predictions = detect_behavior(model, temp_frame_path, is_legacy_model, threshold)
            
            # Print result every 10 frames to avoid console spam
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}: {label} (Confidence: {confidence:.2f})")
            
            # Clean up the temporary frame file immediately
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)

            # Set color based on label
            color_map = {
                "none": (0, 255, 0),  # Green
                "smoking": (0, 0, 255),  # Red
                "alcohol": (255, 0, 0),  # Blue
                "both": (255, 0, 255)  # Magenta
            }
            color = color_map.get(label, (0, 255, 0))
            
            # Draw results on frame
            cv2.putText(display_frame, f"{label}: {confidence:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Add probabilities for all classes (more compact for video)
            y_offset = 60
            for i, class_name in enumerate(CLASSES):
                prob_text = f"{class_name}: {predictions[i]:.2f}"
                cv2.putText(display_frame, prob_text, 
                      (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(display_frame, timestamp, 
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if out:
                out.write(display_frame)

            if not headless:
                cv2.imshow('Behavior Detection', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if out:
            out.release()
        if not headless:
            cv2.destroyAllWindows()
        # Clean up temporary directory
        if os.path.exists('temp_video_frames'):
            import shutil
            shutil.rmtree('temp_video_frames')

def main():
    parser = argparse.ArgumentParser(description='Test Behavior Detection Model (Smoking, Alcohol, Both, or None)')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--video', type=str, help='Path to test video')
    parser.add_argument('--save', action='store_true', help='Save output results')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--model', type=str, default='combined_detector_model_finetuned.keras',
                       help='Path to the model file (default: combined_detector_model_finetuned.keras)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--legacy', action='store_true', 
                        help='Use legacy smoking detection model (for testing before training the new model)')
    args = parser.parse_args()

    # Default to legacy mode with the smoking detector model if the combined model doesn't exist yet
    is_legacy_model = args.legacy
    model_path = args.model
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        if os.path.exists('smoking_detector_model_finetuned.keras'):
            print("Using legacy smoking detector model instead.")
            model_path = 'smoking_detector_model_finetuned.keras'
            is_legacy_model = True
        elif os.path.exists('smoking_detector_model.keras'):
            print("Using legacy smoking detector model instead.")
            model_path = 'smoking_detector_model.keras'
            is_legacy_model = True
        else:
            print("No model files found. Please train a model first.")
            return

    try:
        print(f"Loading model: {model_path}")
        model = load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if args.image:
        print(f"Processing image: {args.image}")
        if not os.path.exists(args.image):
            print(f"Error: Image file not found at {args.image}")
            return
        process_image(model, args.image, args.save, args.headless, is_legacy_model, args.threshold)
    elif args.video:
        print(f"Processing video: {args.video}")
        if not os.path.exists(args.video):
            print(f"Error: Video file not found at {args.video}")
            return
        process_video(model, args.video, args.save, args.headless, is_legacy_model, args.threshold)
    else:
        print("Please provide either --image or --video argument")
        print("Example usage:")
        print("  For image: python test_model.py --image path/to/image.jpg --save")
        print("  For video: python test_model.py --video path/to/video.mp4 --save")
        print("  Add --headless to skip displaying windows for terminal-only output.")
        print("  Add --legacy to use the legacy smoking detection model.")

if __name__ == "__main__":
    main() 