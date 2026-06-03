#!/usr/bin/env python3
"""
Sync Vision — Landmark Classifier Training Script
==================================================
Trains an EfficientNet-Lite0-based landmark classification model and exports as INT8 TFLite.

IMPORTANT: Google provides a PRE-TRAINED landmark TFLite model on Kaggle!
  → https://www.kaggle.com/models/google/landmarks
  → Select "TFLite" → Choose region → Download → save as landmark_effnet_int8.tflite
  → Place in: app/src/main/assets/models/
  This is MUCH easier than training from scratch. Only train if you need custom landmarks.

Dataset Options (for custom training):
  1. Google Landmarks v2 (Kaggle): https://www.kaggle.com/c/landmark-recognition-2021/data
  2. Landmarks dataset (smaller): https://www.kaggle.com/datasets/ekaterinaprokopenko/landmarks
  3. Custom: Create folders named after each landmark with images inside

Usage:
  pip install tensorflow
  python3 train_landmark_model.py --dataset /path/to/landmarks_dataset --output /path/to/sync-vision/app/src/main/assets/models/landmark_effnet_int8.tflite

Expected dataset structure:
  landmarks_dataset/
  ├── Eiffel_Tower/
  │   ├── img001.jpg
  │   └── ...
  ├── Taj_Mahal/
  ├── Colosseum/
  └── ...
"""

import os
import sys
import argparse


def train_landmark_model(dataset_path, output_path, epochs=20, batch_size=32):
    """Train landmark classification model with Keras + EfficientNetB0."""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    IMAGE_SIZE = 224
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Create datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        label_mode='int'
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        label_mode='int'
    )
    
    num_classes = len(train_ds.class_names)
    print(f"Number of landmark classes: {num_classes}")
    print(f"Classes: {train_ds.class_names}")
    
    # Optimize for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Build model with EfficientNetB0 backbone
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # Train
    print("\nTraining...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    
    # Fine-tune
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nFine-tuning...")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs // 2)
    
    # Convert to TFLite with quantization
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    def representative_dataset():
        for images, _ in train_ds.take(100):
            yield [images]
    converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    # Save
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size = os.path.getsize(output_path)
    print(f"\n✓ Model exported to: {output_path}")
    print(f"  Size: {model_size / 1024 / 1024:.1f} MB")
    
    # Also save class names for the Java code
    class_names_path = output_path.replace('.tflite', '_labels.txt')
    with open(class_names_path, 'w') as f:
        for name in train_ds.class_names:
            f.write(name.upper().replace('_', ' ') + '\n')
    print(f"  Labels saved to: {class_names_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Landmark Classification Model')
    parser.add_argument('--dataset', required=True, help='Path to landmarks dataset folder')
    parser.add_argument('--output', default=None, help='Output TFLite model path')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    
    args = parser.parse_args()
    
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        args.output = os.path.join(
            project_dir, 'app', 'src', 'main', 'assets', 'models', 'landmark_effnet_int8.tflite'
        )
    
    train_landmark_model(args.dataset, args.output, args.epochs, args.batch_size)
