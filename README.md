# Smoking and Alcohol Detection System

This project implements a multi-class detection system using computer vision and deep learning. It can detect smoking, alcohol consumption, both behaviors simultaneously, or none of these behaviors in both live video (webcam) and pre-recorded videos/images.

## Features

- Multi-class detection: smoking, alcohol consumption, both, or none
- Real-time detection using webcam
- Image and video file processing
- Transfer learning with MobileNetV2 architecture
- Confidence score display for all classes
- Enhanced processing for better accuracy in difficult lighting conditions

## Requirements

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be organized in the following structure:
```
dataset/
├── Training/
│   ├── none/
│   ├── smoking/
│   ├── alcohol/
│   └── both/
├── Validation/
│   ├── none/
│   ├── smoking/
│   ├── alcohol/
│   └── both/
└── Testing/
    ├── none/
    ├── smoking/
    ├── alcohol/
    └── both/
```

## Setting Up the Dataset

### Option 1: Download the Alcohol Detection Dataset
You can download an alcohol detection dataset from Roboflow using the provided script:

```bash
python download_alcohol_dataset.py --key YOUR_ROBOFLOW_API_KEY
```

This script will:
1. Download an alcohol detection dataset
2. Organize it into the correct folder structure
3. Create the 'both' class by combining smoking and alcohol images
4. Rename the 'not smoking' directory to 'none'

### Option 2: Manual Setup
1. Create the directory structure as shown above
2. Place smoking images in the 'smoking' directories
3. Place alcohol consumption images in the 'alcohol' directories
4. Place images with both behaviors in the 'both' directories
5. Place images with neither behavior in the 'none' directories

## Usage

1. First, train the multi-class model:
```bash
python train_model.py
```

2. For detection in an image:
```bash
python detect_behavior.py --image path/to/image.jpg
```

3. For live detection using webcam:
```bash
python detect_behavior.py
```

4. For detection in a video file:
```bash
python detect_behavior.py --video path/to/video.mp4
```

5. For testing the model on individual images:
```bash
python test_model.py --image path/to/image.jpg --save
```

## Additional Options

- `--threshold`: Set a custom confidence threshold (default varies by image format)
- `--sensitive`: Use a lower threshold (0.15) for more sensitive detection
- `--save-all`: Save all processed versions of an image for comparison
- `--no-save`: Disable saving output (saving is enabled by default)
- `--model`: Specify a custom model file path

## Controls

- Press 'q' to quit the video stream
- The detection results will be displayed in real-time with confidence scores for all classes

## How It Works

1. The system uses transfer learning with MobileNetV2 pretrained on ImageNet.
2. Images are processed with multiple enhancement techniques to improve detection accuracy.
3. For each image, the system tries different processing methods (original, brightness adjustment, contrast adjustment, enhanced) and selects the one with the highest confidence.
4. The model outputs probabilities for all four classes (none, smoking, alcohol, both).

## Notes

- The system works best with clear, well-lit video.
- WEBP images are processed with special handling to improve detection accuracy.
- For optimal results, ensure the subject's face and upper body are visible in the frame.
- The model is trained to recognize both the action and objects related to smoking and drinking. 

## Recent Updates - Improved Smoking Detection

The smoking detection algorithm has been significantly improved to reduce false positives:

1. **Enhanced Color Detection**: The color detection for cigarette glow now uses more specific HSV ranges with higher saturation and value thresholds.

2. **Improved Shape Analysis**: 
   - Added complexity calculation for contour shapes
   - Tightened aspect ratio requirements for cigarette detection
   - Added area constraints to filter out irrelevant shapes
   - Increased the Gaussian blur kernel size for better noise reduction

3. **Stricter Classification Thresholds**:
   - Increased minimum confidence thresholds across all detection methods
   - Enhanced temporal smoothing to require more consistent detections across frames
   - Reduced maximum confidence values to prevent overconfident classifications

4. **Multi-criteria Validation**:
   - Now requires multiple positive signals from different detection methods for high-confidence classifications
   - Added shape complexity analysis to filter out objects that resemble cigarettes but have irregular shapes

5. **Hand-to-Mouth Gesture Detection**:
   - Added skin tone detection to identify potential hand regions
   - Implemented proximity analysis between cigarette contours and hand contours
   - Increased confidence scores for cigarettes detected near hands (typical smoking gesture)
   - Visual highlighting of hand-to-cigarette connections in the processed frames

6. **Portrait Mode Detection**:
   - Added facial detection to identify portrait-style images
   - Implemented special handling for portrait photos with stricter criteria
   - Requires multiple strong smoking indicators for positive detection in portraits
   - Adjusts detection thresholds based on face size relative to the image
   - Shows "PORTRAIT MODE" indicator with higher thresholds for face-centric images
   - Prevents false positives in portrait photos while maintaining accurate detection in smoking scenarios

These improvements have significantly reduced false positive detections while maintaining sensitivity to actual smoking behavior in both static images and real-time video.

## Troubleshooting

If you encounter incorrect detections:

1. **For Portrait Photos**: The system now has special handling for portrait-style photos. If you're still getting false positives, try adjusting the face detection threshold in the code.

2. **For Smoking Detection**: If genuine smoking isn't being detected, you can adjust the detection thresholds in `realtime_detector.py`. Look for parameters like `orange_threshold`, `min_contours`, and `detection_threshold`.

3. **For Face Detection Issues**: Make sure the OpenCV Haar cascades are installed correctly. The system uses `haarcascade_frontalface_default.xml` for face detection.