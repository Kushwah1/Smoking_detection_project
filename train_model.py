import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import to_categorical
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import pathlib
import shutil

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 100  # Increase epochs
AUTOTUNE = tf.data.AUTOTUNE

# Define the classes
CLASSES = ["none", "smoking", "alcohol", "both"]
NUM_CLASSES = len(CLASSES)

def process_path(file_path):
    # Add error handling for image decoding
    try:
        # Load the image
        img = tf.io.read_file(file_path)
        
        # Try to decode with multiple methods
        try:
            # First try decode_image with error ignoring
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
        except:
            # Fall back to more specific decoders
            try:
                # Try JPEG decoder
                img = tf.image.decode_jpeg(img, channels=3)
            except:
                try:
                    # Try PNG decoder
                    img = tf.image.decode_png(img, channels=3)
                except:
                    # Last resort - return zeros
                    tf.print(f"Failed to decode image: {file_path}")
                    return tf.zeros(IMAGE_SIZE + (3,), dtype=tf.float32), tf.constant(0, dtype=tf.int32)
        
        # Check if image was successfully decoded and has valid dimensions
        if tf.logical_or(
            tf.reduce_sum(tf.cast(tf.math.is_nan(img), tf.int32)) > 0,
            tf.logical_or(tf.shape(img)[0] == 0, tf.shape(img)[1] == 0)
        ):
            tf.print(f"Warning: Image possibly corrupted or has zero dimensions: {file_path}")
            return tf.zeros(IMAGE_SIZE + (3,), dtype=tf.float32), tf.constant(0, dtype=tf.int32)
        
        # Resize image
        img = tf.image.resize(img, IMAGE_SIZE)
        # Normalize pixel values
        img = tf.cast(img, tf.float32) / 255.0
        
        # Get the label from the path
        parts = tf.strings.split(file_path, os.path.sep)[-2]
        
        # Map folder names to class indices
        label = tf.constant(0, dtype=tf.int32)  # default to "none"
        
        conditions = [
            tf.strings.regex_full_match(parts, "none"),
            tf.strings.regex_full_match(parts, "smoking"),
            tf.strings.regex_full_match(parts, "alcohol"),
            tf.strings.regex_full_match(parts, "both")
        ]
        
        values = [0, 1, 2, 3]  # Corresponding indices
        label = tf.case([(conditions[i], lambda i=i: tf.constant(values[i], dtype=tf.int32)) for i in range(len(conditions))])
        
        return img, label
    except Exception as e:
        tf.print(f"Error processing image {file_path}: {e}")
        return tf.zeros(IMAGE_SIZE + (3,), dtype=tf.float32), tf.constant(0, dtype=tf.int32)

def configure_for_performance(dataset, is_training=False):
    dataset = dataset.cache()
    if is_training:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def data_augmentation(image):
    # More aggressive data augmentation
    
    # Random flip
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Random rotation (more angles)
    image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    
    # Random brightness (more range)
    image = tf.image.random_brightness(image, 0.3)
    
    # Random contrast (more range)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    
    # Random hue and saturation
    image = tf.image.random_hue(image, 0.2)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    
    return image

def prepare_dataset(data_dir, is_training=False):
    data_dir = pathlib.Path(data_dir)
    
    # Get list of image files, including different extensions
    image_patterns = [
        str(data_dir/'*/*.jpg'),
        str(data_dir/'*/*.jpeg'),
        str(data_dir/'*/*.png'),
        str(data_dir/'*/*.webp')  # Added WEBP support
    ]
    
    # More robust file listing with error handling
    try:
        image_files = []
        for pattern in image_patterns:
            try:
                files = tf.data.Dataset.list_files(pattern, shuffle=is_training)
                image_files.append(files)
            except tf.errors.InvalidArgumentError:
                print(f"Warning: No files found matching pattern {pattern}")
        
        if not image_files:
            raise ValueError(f"No image files found in {data_dir}")
        
        # Combine multiple datasets if we have them
        if len(image_files) > 1:
            image_dataset = image_files[0]
            for files in image_files[1:]:
                image_dataset = image_dataset.concatenate(files)
        else:
            image_dataset = image_files[0]
    except Exception as e:
        print(f"Error listing files: {e}")
        raise
    
    # Print number of files found
    file_count = tf.data.experimental.cardinality(image_dataset).numpy()
    print(f"Found {file_count} images in {data_dir}")
    
    # Create dataset
    dataset = image_dataset.map(process_path, num_parallel_calls=AUTOTUNE)
    
    # Filter out invalid images (zeros from error handling)
    def is_valid_image(image, label):
        return tf.reduce_sum(image) > 0
    
    dataset = dataset.filter(is_valid_image)
    
    # Check if we still have images after filtering
    filtered_count = tf.data.experimental.cardinality(dataset).numpy()
    if filtered_count == 0:
        raise ValueError(f"No valid images found in {data_dir} after filtering corrupted files")
    
    print(f"Using {filtered_count} valid images after filtering")
    
    if is_training:
        # Apply augmentation only to training data
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x), y),
            num_parallel_calls=AUTOTUNE
        )
    
    return configure_for_performance(dataset, is_training)

def create_model():
    # Use pretrained MobileNetV2 as a feature extractor
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Create model with pretrained base
    model = Sequential([
        # Base pretrained model
        base_model,
        
        # Add new classifier layers
        GlobalAveragePooling2D(),
        BatchNormalization(),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # For multi-class classification (none, smoking, alcohol, both)
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Very low learning rate for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

def compute_class_weights(dataset):
    # Count samples in each class
    class_counts = np.zeros(NUM_CLASSES, dtype=np.int32)
    
    # Iterate through the dataset to count samples in each class
    for _, labels in dataset:
        # Convert one-hot to indices if needed
        for label in labels:
            class_counts[label.numpy()] += 1
    
    total = np.sum(class_counts)
    
    # Compute class weights
    class_weights = {}
    for i in range(NUM_CLASSES):
        # Add small epsilon to avoid division by zero
        class_weights[i] = (total / NUM_CLASSES) / (class_counts[i] + 1e-6)
    
    print(f"Class distribution - {CLASSES[0]}: {class_counts[0]}, "
          f"{CLASSES[1]}: {class_counts[1]}, "
          f"{CLASSES[2]}: {class_counts[2]}, "
          f"{CLASSES[3]}: {class_counts[3]}")
    
    print(f"Class weights - {CLASSES[0]}: {class_weights[0]:.2f}, "
          f"{CLASSES[1]}: {class_weights[1]:.2f}, "
          f"{CLASSES[2]}: {class_weights[2]:.2f}, "
          f"{CLASSES[3]}: {class_weights[3]:.2f}")
    
    return class_weights

def find_corrupted_images(data_dir):
    """Utility function to find and report corrupted images in a directory"""
    print(f"Checking for corrupted images in {data_dir}...")
    corrupted_images = []
    
    # Get all image files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                file_path = os.path.join(root, file)
                try:
                    # Try to read and decode the image
                    with open(file_path, 'rb') as f:
                        img_data = f.read()
                    
                    # Try with PIL
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(img_data))
                    img.verify()  # verify that it is an image
                except Exception as e:
                    print(f"Found corrupted image: {file_path}")
                    print(f"  Error: {str(e)}")
                    corrupted_images.append(file_path)
    
    if corrupted_images:
        print(f"Found {len(corrupted_images)} corrupted images:")
        for img in corrupted_images:
            print(f"  {img}")
    else:
        print("No corrupted images found.")
    
    return corrupted_images

def verify_dataset():
    """Verify if the dataset structure is correct with all required folders"""
    required_dirs = ['dataset/Training', 'dataset/Validation']
    
    # Check if main directories exist
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"Error: Directory {dir_path} does not exist!")
            return False
        
        # Check if class subdirectories exist
        for subdir in CLASSES:
            full_path = os.path.join(dir_path, subdir)
            if not os.path.exists(full_path):
                print(f"Error: Class directory {full_path} does not exist!")
                # Create the directory if it doesn't exist
                os.makedirs(full_path, exist_ok=True)
                print(f"Created directory {full_path}")
    
    # Check for corrupted images
    corrupted_train = find_corrupted_images('dataset/Training')
    corrupted_val = find_corrupted_images('dataset/Validation')
    
    if corrupted_train or corrupted_val:
        print("Warning: Corrupted images found. They will be skipped during training.")
    
    return True

def setup_combined_dataset():
    """Setup the dataset directory structure for combined detection"""
    # Check if we need to reorganize the dataset
    if os.path.exists('dataset/Training/none') and os.path.exists('dataset/Training/both'):
        print("Dataset already set up for combined detection.")
        return True
    
    # Create required directories if they don't exist
    for split in ['Training', 'Validation', 'Testing']:
        for class_name in CLASSES:
            os.makedirs(f'dataset/{split}/{class_name}', exist_ok=True)
    
    # If smoking and alcohol directories already exist, we need to reorganize
    if os.path.exists('dataset/Training/smoking') and not os.path.exists('dataset/Training/alcohol'):
        print("Setting up placeholder directories for alcohol detection...")
        print("You'll need to add alcohol detection images to these directories.")
        return True
    
    # If we had older "not smoking" directory, rename to "none"
    for split in ['Training', 'Validation', 'Testing']:
        old_dir = f'dataset/{split}/not smoking'
        new_dir = f'dataset/{split}/none'
        if os.path.exists(old_dir) and not os.path.exists(new_dir):
            print(f"Renaming '{old_dir}' to '{new_dir}'")
            os.rename(old_dir, new_dir)
    
    print("Dataset directory structure set up for combined detection.")
    return True

def plot_training_history(history, save_dir='training_plots'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot precision and recall
    plt.subplot(1, 3, 3)
    plt.plot(history.history['precision'], label='Precision')
    plt.plot(history.history['recall'], label='Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history_{timestamp}.png')
    plt.close()

def train_model():
    # Setup dataset structure for combined detection
    if not setup_combined_dataset():
        print("Error setting up the dataset structure.")
        sys.exit(1)
    
    # Verify dataset
    if not verify_dataset():
        print("Please check your dataset structure and try again.")
        sys.exit(1)
    
    try:
        print("Preparing datasets...")
        train_dataset = prepare_dataset('dataset/Training', is_training=True)
        val_dataset = prepare_dataset('dataset/Validation', is_training=False)
        
        # Compute class weights to handle imbalance
        class_weights = compute_class_weights(train_dataset)
        
        print("Creating model with MobileNetV2 backbone...")
        model = create_model()
        
        # Remove early stopping to ensure full training
        print(f"Starting model training for {EPOCHS} epochs...")
        history = model.fit(
            train_dataset,
            epochs=EPOCHS,
            validation_data=val_dataset,
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=7,
                    min_lr=1e-6
                )
            ]
        )

        # Save the model
        model.save('combined_detector_model.keras')
        print("Model saved successfully in Keras format!")
        
        # Print final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        final_precision = history.history['precision'][-1]
        final_recall = history.history['recall'][-1]
        
        print(f"\nFinal Training Accuracy: {final_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_val_acc:.4f}")
        print(f"Final Precision: {final_precision:.4f}")
        print(f"Final Recall: {final_recall:.4f}")
        
        # Create and save training plots
        plot_training_history(history)
        print("Training plots saved in 'training_plots' directory")
        
        # After initial training, unfreeze some layers and fine-tune
        print("\nFine-tuning the model...")
        
        # Unfreeze the top layers of the base model
        base_model = model.layers[0]
        base_model.trainable = True
        
        # Freeze all the layers except the top 10
        for layer in base_model.layers[:-10]:
            layer.trainable = False
            
        # Compile the model with a very low learning rate
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        # Fine-tune the model
        fine_tune_history = model.fit(
            train_dataset,
            epochs=20,  # additional 20 epochs for fine-tuning
            validation_data=val_dataset,
            class_weight=class_weights,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-7
                )
            ]
        )
        
        # Save the fine-tuned model
        model.save('combined_detector_model_finetuned.keras')
        print("Fine-tuned model saved successfully!")
        
        # Print final metrics after fine-tuning
        final_ft_train_acc = fine_tune_history.history['accuracy'][-1]
        final_ft_val_acc = fine_tune_history.history['val_accuracy'][-1]
        final_ft_precision = fine_tune_history.history['precision'][-1]
        final_ft_recall = fine_tune_history.history['recall'][-1]
        
        print(f"\nAfter Fine-tuning:")
        print(f"Final Training Accuracy: {final_ft_train_acc:.4f}")
        print(f"Final Validation Accuracy: {final_ft_val_acc:.4f}")
        print(f"Final Precision: {final_ft_precision:.4f}")
        print(f"Final Recall: {final_ft_recall:.4f}")
        
        # Create and save fine-tuning plots
        plot_training_history(fine_tune_history, save_dir='finetune_plots')
        print("Fine-tuning plots saved in 'finetune_plots' directory")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    train_model() 