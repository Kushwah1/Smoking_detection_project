import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt

print("Libraries imported successfully")

# Constants
IMAGE_SIZE = (224, 224)
CLASSES = ["none", "smoking", "alcohol", "both"]

def preprocess_image(img_path):
    """Load and preprocess a single image"""
    try:
        print(f"Opening image {img_path}")
        img = Image.open(img_path).convert('RGB')
        print(f"Resizing image to {IMAGE_SIZE}")
        img = img.resize(IMAGE_SIZE)
        print("Converting to numpy array")
        img_array = np.array(img, dtype=np.float32) / 255.0
        print("Preprocessing complete")
        return img_array
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def test_image(img_path):
    """Test an image with the trained model"""
    print("\n" + "="*50)
    print(f"Starting test_image function with {img_path}")
    print("="*50 + "\n")
    
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return
    else:
        print(f"Image found at {img_path}")
    
    # Try different model files
    model_files = [
        'smoking_detector_model.h5',
        'smoking_detector_model.keras',
        'smoking_detector_model_finetuned.keras',
        'combined_detector_model.keras',
        'combined_detector_model_finetuned.keras'
    ]
    
    model = None
    for model_path in model_files:
        if os.path.exists(model_path):
            print(f"Trying to load model from {model_path}...")
            try:
                # Use load_model with custom objects to handle compatibility issues
                model = tf.keras.models.load_model(model_path, compile=False)
                print(f"Model loaded successfully from {model_path}!")
                break
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
        else:
            print(f"Model file {model_path} not found.")
    
    if model is None:
        print("Could not load any model file. Please train the model first.")
        return
    
    # Preprocess image
    print(f"Processing image: {img_path}")
    img_array = preprocess_image(img_path)
    if img_array is None:
        return
    
    # Add batch dimension
    print("Adding batch dimension")
    img_batch = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    print("Making prediction...")
    try:
        predictions = model.predict(img_batch, verbose=1)
        print("Prediction complete")
        
        # Handle different output formats
        if len(predictions.shape) == 2:
            # Standard multi-class output
            pred_array = predictions[0]
        else:
            # Single output (binary classification)
            print("Converting binary output to multi-class format")
            smoking_prob = float(predictions[0][0])
            pred_array = np.array([1.0 - smoking_prob, smoking_prob, 0.0, 0.0])
            
        predicted_class_idx = np.argmax(pred_array)
        predicted_class = CLASSES[predicted_class_idx]
        confidence = pred_array[predicted_class_idx] * 100
        
        # Print results
        print("\nPrediction Results:")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        
        # Show all class probabilities
        print("\nClass Probabilities:")
        for i, class_name in enumerate(CLASSES):
            if i < len(pred_array):
                print(f"{class_name}: {pred_array[i]*100:.2f}%")
        
        # Display the image with prediction
        print("Creating visualization...")
        plt.figure(figsize=(8, 8))
        img = Image.open(img_path).convert('RGB')
        plt.imshow(img)
        plt.title(f"Prediction: {predicted_class} ({confidence:.2f}%)")
        plt.axis('off')
        
        # Save the figure
        output_path = f"detection_results/{os.path.splitext(os.path.basename(img_path))[0]}_prediction.png"
        os.makedirs("detection_results", exist_ok=True)
        plt.savefig(output_path)
        print(f"Prediction image saved to {output_path}")
        
        # Show the plot
        try:
            print("Attempting to display plot...")
            plt.show()
        except Exception as e:
            print(f"Could not display interactive plot: {e}")
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    print("Test complete!")

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Get image path from command line or use the specified path
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        print(f"Using image path from command line: {img_path}")
    else:
        img_path = "C:\\Users\\Gaurav\\Downloads\\project_minor\\woman-drinking-beer-alone-at-the-bar-header-1024x575.webp"
        print(f"Using default image path: {img_path}")
    
    test_image(img_path) 