import cv2
import numpy as np
import os
import sys
import time
import glob
from PIL import Image, ImageEnhance
import argparse
import datetime

# Import detection function from realtime_detector
from realtime_detector import detect_smoking, enhance_frame

def process_single_image(image_path, output_dir='detection_results'):
    """Process a single image for smoking detection"""
    print(f"\nProcessing image: {image_path}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        return False
    
    # Load the image
    try:
        img = cv2.imread(image_path)
        if img is None:
            # Try with PIL
            print("OpenCV couldn't read image, trying with PIL...")
            pil_img = Image.open(image_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error loading image: {e}")
        return False
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Make a copy for display
    display_img = img.copy()
    
    # Enhance the image for better detection
    enhanced_img = enhance_frame(img)
    
    # Detect smoking
    print("Running smoking detection...")
    t_start = time.time()
    label, confidence, processed_img = detect_smoking(enhanced_img)
    t_elapsed = time.time() - t_start
    print(f"Detection completed in {t_elapsed:.2f} seconds")
    
    # Print results
    print(f"Results: {label.upper()} with {confidence:.2f} confidence")
    
    # Create color map for different classes
    color_map = {
        "not_smoking": (0, 255, 0),  # Green
        "smoking": (0, 0, 255),      # Red
    }
    color = color_map.get(label, (255, 255, 255))
    
    # Add results to image
    cv2.putText(
        display_img, 
        f"{label.upper()}: {confidence:.2f}", 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        color, 
        2
    )
    
    # Add processing time
    cv2.putText(
        display_img, 
        f"Processing time: {t_elapsed:.2f}s", 
        (10, 60), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.6, 
        (255, 255, 255), 
        1
    )
    
    # Add timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(
        display_img, 
        timestamp, 
        (width - 200, height - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        (255, 255, 255), 
        1
    )
    
    # Combine the original (with text) and processed image (with contours)
    alpha = 0.3  # Transparency factor
    overlay = cv2.addWeighted(display_img, 1 - alpha, processed_img, alpha, 0)
    
    # Draw a colored border based on detection
    border_thickness = 5
    cv2.rectangle(
        overlay, 
        (0, 0), 
        (width-1, height-1), 
        color, 
        border_thickness
    )
    
    # Save the result
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_result_{label}_{confidence:.2f}_{timestamp_str}.jpg")
    cv2.imwrite(output_path, overlay)
    print(f"Result saved as: {output_path}")
    
    # Show the result
    try:
        cv2.imshow("Detection Result", overlay)
        print("Displaying result (press any key to continue)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Could not display image: {e}")
    
    return True

def process_directory(dir_path, output_dir='detection_results'):
    """Process all images in a directory"""
    print(f"Processing all images in directory: {dir_path}")
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
        pattern = os.path.join(dir_path, ext)
        image_files.extend(glob.glob(pattern))
    
    # Process each image
    if not image_files:
        print(f"No image files found in {dir_path}")
        return False
    
    print(f"Found {len(image_files)} image files")
    success_count = 0
    
    for img_path in image_files:
        if process_single_image(img_path, output_dir):
            success_count += 1
    
    print(f"\nProcessed {success_count} out of {len(image_files)} images successfully")
    return True

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test smoking detection on static images")
    parser.add_argument("--image", type=str, help="Path to a single image file")
    parser.add_argument("--dir", type=str, help="Path to directory containing images")
    parser.add_argument("--output", type=str, default="detection_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    if not args.image and not args.dir:
        parser.print_help()
        print("\nError: Please specify either --image or --dir")
        return
    
    # Process single image
    if args.image:
        process_single_image(args.image, args.output)
    
    # Process directory
    if args.dir:
        process_directory(args.dir, args.output)

if __name__ == "__main__":
    main() 