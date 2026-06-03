#!/usr/bin/env python3
"""
Sync Vision — Weather Classifier Training Script
=================================================
Trains a MobileNetV2-based weather classification model and exports as INT8 TFLite.

Classes: CLEAR, PARTLY_CLOUDY, CLOUDY, OVERCAST, RAINY, FOGGY, SUNSET_SUNRISE, STORMY

Dataset Options:
  1. Multi-class Weather Dataset (MWD): https://data.mendeley.com/datasets/4drtyfjtfy/1
  2. Weather Image Recognition (Kaggle): https://www.kaggle.com/datasets/jehanbhathena/weather-dataset
  3. Custom: Create folders named after each class with weather images inside

Usage:
  pip install tensorflow tflite-model-maker
  python3 train_weather_model.py --dataset /path/to/weather_dataset --output /path/to/sync-vision/app/src/main/assets/models/weather_classifier_int8.tflite

Expected dataset structure:
  weather_dataset/
  ├── CLEAR/
  │   ├── img001.jpg
  │   ├── img002.jpg
  │   └── ...
  ├── PARTLY_CLOUDY/
  ├── CLOUDY/
  ├── OVERCAST/
  ├── RAINY/
  ├── FOGGY/
  ├── SUNSET_SUNRISE/
  └── STORMY/
"""

import os
import sys
import argparse

def train_weather_model(dataset_path, output_path, epochs=20, batch_size=32):
    """Train weather classification model using TFLite Model Maker."""
    
    try:
        from tflite_model_maker import image_classifier
        from tflite_model_maker import ImageClassifierDataLoader
        import tensorflow as tf
    except ImportError:
        print("ERROR: tflite-model-maker not installed.")
        print("Install with: pip install tflite-model-maker")
        sys.exit(1)
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Load dataset from folder structure
    data = ImageClassifierDataLoader.from_folder(dataset_path)
    
    # Split into train/validation/test
    train_data, rest_data = data.split(0.8)
    validation_data, test_data = rest_data.split(0.5)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(validation_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Classes: {train_data.num_classes}")
    
    # Create and train model
    model = image_classifier.create(
        train_data,
        model_spec='mobilenet_v2_spec',
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(test_data)
    print(f"\nTest accuracy: {accuracy:.2%}")
    print(f"Test loss: {loss:.4f}")
    
    # Export as INT8 quantized TFLite
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    model.export(
        export_dir=output_dir,
        tflite_filename=os.path.basename(output_path),
        quantization_config=tf.lite.Optimize.DEFAULT
    )
    
    model_size = os.path.getsize(output_path)
    print(f"\n✓ Model exported to: {output_path}")
    print(f"  Size: {model_size / 1024 / 1024:.1f} MB")


def train_weather_model_manual(dataset_path, output_path, epochs=20, batch_size=32):
    """Train weather classification model manually with Keras (no Model Maker)."""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    
    IMAGE_SIZE = 224
    NUM_CLASSES = 8
    
    CLASS_NAMES = [
        'CLEAR', 'PARTLY_CLOUDY', 'CLOUDY', 'OVERCAST',
        'RAINY', 'FOGGY', 'SUNSET_SUNRISE', 'STORMY'
    ]
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Create training dataset
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
    
    # Optimize for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Build model with MobileNetV2 backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
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
    
    # Fine-tune (unfreeze top layers)
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nFine-tuning...")
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs // 2
    )
    
    # Convert to TFLite with INT8 quantization
    print("\nConverting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Representative dataset for INT8 quantization
    def representative_dataset():
        for images, _ in train_ds.take(100):
            yield [images]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    model_size = os.path.getsize(output_path)
    print(f"\n✓ Model exported to: {output_path}")
    print(f"  Size: {model_size / 1024 / 1024:.1f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Weather Classification Model')
    parser.add_argument('--dataset', required=True, help='Path to weather dataset folder')
    parser.add_argument('--output', default=None, help='Output TFLite model path')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    parser.add_argument('--manual', action='store_true', help='Use manual Keras training (no Model Maker)')
    
    args = parser.parse_args()
    
    if args.output is None:
        # Default output path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        args.output = os.path.join(
            project_dir, 'app', 'src', 'main', 'assets', 'models', 'weather_classifier_int8.tflite'
        )
    
    if args.manual:
        train_weather_model_manual(args.dataset, args.output, args.epochs, args.batch_size)
    else:
        train_weather_model(args.dataset, args.output, args.epochs, args.batch_size)
