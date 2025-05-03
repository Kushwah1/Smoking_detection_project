import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from datetime import datetime
import random
import shutil
from PIL import Image

# Constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15  # Reduced for faster training
MODEL_PATH = 'combined_detector_model.keras'
CLASSES = ["none", "smoking", "alcohol", "both"]
NUM_CLASSES = len(CLASSES)

def ensure_class_directories():
    """Make sure all class directories exist"""
    for split in ['Training', 'Validation']:
        for class_name in CLASSES:
            os.makedirs(f"dataset/{split}/{class_name}", exist_ok=True)
    print("Class directories verified.")

def preprocess_image(img_path):
    """Load and preprocess a single image"""
    try:
        # Use PIL for consistent loading
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def load_dataset(dataset_dir):
    """Load dataset using simple Python/PIL approach instead of TF data pipeline"""
    images = []
    labels = []
    class_counts = {cls: 0 for cls in CLASSES}
    
    # Go through each class directory
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        print(f"Loading {class_name} images from {class_dir}...")
        image_files = [f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        # If directory is empty, try to generate some samples
        if len(image_files) == 0:
            if class_name in ['alcohol', 'both'] and os.path.exists(os.path.join(dataset_dir, 'smoking')):
                print(f"No images found in {class_dir}, generating samples from smoking class...")
                create_synthetic_samples(dataset_dir, class_name)
                image_files = [f for f in os.listdir(class_dir) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        # Process each image in the class directory
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            img_array = preprocess_image(img_path)
            
            if img_array is not None:
                images.append(img_array)
                labels.append(class_idx)
                class_counts[class_name] += 1
    
    # Print class distribution
    print(f"Class distribution in {dataset_dir}:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} images")
    
    # Convert to numpy arrays
    if len(images) == 0:
        print(f"ERROR: No valid images found in {dataset_dir}")
        return None, None
    
    X = np.array(images)
    y = np.array(labels)
    
    return X, y

def create_synthetic_samples(dataset_dir, target_class, num_samples=10):
    """Create synthetic samples for empty classes"""
    smoking_dir = os.path.join(dataset_dir, 'smoking')
    target_dir = os.path.join(dataset_dir, target_class)
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    
    # Get smoking images
    smoking_images = [f for f in os.listdir(smoking_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not smoking_images:
        print(f"No smoking images found to create {target_class} samples")
        return
    
    # Select a subset of smoking images
    selected_images = random.sample(smoking_images, min(num_samples, len(smoking_images)))
    
    for i, img_name in enumerate(selected_images):
        src_path = os.path.join(smoking_dir, img_name)
        try:
            img = Image.open(src_path).convert("RGB")
            
            if target_class == "alcohol":
                # Apply red tint for alcohol
                r, g, b = img.split()
                r_array = np.array(r)
                r_array = np.clip(r_array.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
                r = Image.fromarray(r_array)
                modified = Image.merge("RGB", (r, g, b))
                
            elif target_class == "both":
                # Combine modifications for both
                r, g, b = img.split()
                r_array = np.array(r)
                r_array = np.clip(r_array.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
                r = Image.fromarray(r_array)
                modified = Image.merge("RGB", (r, g, b))
                
                # Add a visual indicator (like a horizontal bar)
                modified_array = np.array(modified)
                height, width = modified_array.shape[:2]
                bar_height = height // 10
                bar_y = height - bar_height - 10
                modified_array[bar_y:bar_y+bar_height, :, 0] = 255  # Red bar
                modified = Image.fromarray(modified_array)
            
            # Save the modified image
            dst_path = os.path.join(target_dir, f"{target_class}_{i+1}_{img_name}")
            modified.save(dst_path)
            print(f"Created {target_class} sample: {dst_path}")
            
        except Exception as e:
            print(f"Error creating sample from {img_name}: {e}")

def data_augmentation(X, y):
    """Perform simple data augmentation to balance classes"""
    class_counts = {}
    for label in y:
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1
    
    # Find the class with the most samples
    max_samples = max(class_counts.values()) if class_counts else 0
    
    if max_samples == 0:
        return X, y
    
    # Augment each class to have roughly the same number of samples
    augmented_images = []
    augmented_labels = []
    
    for class_idx in range(NUM_CLASSES):
        # Get indices of this class
        indices = np.where(y == class_idx)[0]
        
        if len(indices) == 0:
            continue
        
        # Add all original samples
        class_images = X[indices]
        augmented_images.append(class_images)
        augmented_labels.extend([class_idx] * len(indices))
        
        # Augment if needed
        if len(indices) < max_samples:
            # Calculate how many more samples we need
            num_to_augment = max_samples - len(indices)
            
            # Simple augmentation by randomly flipping and rotating existing images
            for i in range(num_to_augment):
                idx = indices[i % len(indices)]  # Cycle through available images
                img = X[idx].copy()
                
                # Random horizontal flip
                if random.random() > 0.5:
                    img = np.fliplr(img)
                
                # Random brightness adjustment
                img = img * random.uniform(0.8, 1.2)
                img = np.clip(img, 0, 1.0)
                
                augmented_images.append(img)
                augmented_labels.append(class_idx)
    
    # Combine all augmented images
    X_aug = np.vstack(augmented_images) if augmented_images else np.array([])
    y_aug = np.array(augmented_labels)
    
    # Shuffle the data
    indices = np.arange(len(y_aug))
    np.random.shuffle(indices)
    X_aug = X_aug[indices]
    y_aug = y_aug[indices]
    
    return X_aug, y_aug

def create_model():
    """Create model architecture using MobileNetV2 backbone"""
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
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # For multi-class classification
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    # Compile with a reasonable learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

def plot_training_history(history, save_dir='training_plots'):
    """Plot and save training metrics history"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(15, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot precision and recall
    plt.subplot(1, 3, 3)
    if 'precision' in history.history:
        plt.plot(history.history['precision'], label='Precision')
    if 'recall' in history.history:
        plt.plot(history.history['recall'], label='Recall')
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_history_{timestamp}.png')
    plt.close()

def copy_test_samples():
    """Make sure we have test samples for each class"""
    for class_name in CLASSES:
        test_dir = f"dataset/Testing/{class_name}"
        train_dir = f"dataset/Training/{class_name}"
        
        # Create test directory if it doesn't exist
        os.makedirs(test_dir, exist_ok=True)
        
        # If test directory is empty but training has samples, copy one over
        if len(os.listdir(test_dir)) == 0 and os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0:
            print(f"Copying sample from {train_dir} to {test_dir}")
            train_files = [f for f in os.listdir(train_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if train_files:
                src_file = os.path.join(train_dir, train_files[0])
                dst_file = os.path.join(test_dir, "sample.jpg")
                shutil.copy(src_file, dst_file)

def main():
    """Main function to train the model"""
    print("Simple Combined Behavior Detection Training")
    print("===========================================")
    
    # Make sure directories exist
    ensure_class_directories()
    
    # Load training data
    print("\nLoading training data...")
    X_train, y_train = load_dataset('dataset/Training')
    
    if X_train is None or len(X_train) == 0:
        print("ERROR: No training data available. Exiting.")
        return
    
    # Load validation data
    print("\nLoading validation data...")
    X_val, y_val = load_dataset('dataset/Validation')
    
    # If no validation data, use a portion of training data
    if X_val is None or len(X_val) == 0:
        print("No separate validation data, using 20% of training data for validation.")
        
        # Split training data
        split_idx = int(len(X_train) * 0.8)
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
    
    # Augment training data for better balance
    print("\nAugmenting training data to balance classes...")
    X_train, y_train = data_augmentation(X_train, y_train)
    
    # Create and compile model
    print("\nCreating and compiling model...")
    model = create_model()
    model.summary()
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    
    # Save the model
    print("\nSaving model...")
    model.save('combined_detector_model.keras')
    
    # Plot and save training history
    plot_training_history(history)
    
    # Fine-tune the model
    print("\nFine-tuning model...")
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze all except the last few layers
    for layer in base_model.layers[:-5]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Fine-tune
    fine_tune_history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,  # Fewer epochs for fine-tuning
        batch_size=BATCH_SIZE
    )
    
    # Save fine-tuned model
    model.save('combined_detector_model_finetuned.keras')
    
    # Plot fine-tuning history
    plot_training_history(fine_tune_history, save_dir='finetune_plots')
    
    # Make sure we have test samples
    copy_test_samples()
    
    print("\nTraining completed! You can now use detect_behavior.py to test the model.")

if __name__ == "__main__":
    main() 