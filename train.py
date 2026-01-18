"""
Traffic Sign Classification - Training Script
==============================================

This script trains a CNN model for traffic sign classification using the GTSRB dataset.
It uses transfer learning with MobileNetV2 for efficient and accurate classification.

Usage:
    python train.py --epochs 20 --batch_size 128 --img_size 48
"""

import os
import argparse
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Traffic Sign Classifier')
    
    parser.add_argument('--data_dir', type=str, default='data/Train',
                        help='Path to training data directory')
    parser.add_argument('--model_dir', type=str, default='model',
                        help='Directory to save trained model')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=48,
                        help='Image size (height and width)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    
    return parser.parse_args()


def create_data_generators(data_dir, img_size, batch_size, val_split):
    """
    Create training and validation data generators with augmentation
    
    Args:
        data_dir: Path to training data directory
        img_size: Target image size
        batch_size: Batch size
        val_split: Validation split ratio
    
    Returns:
        train_generator, val_generator
    """
    print(f"\n📁 Loading data from: {data_dir}")
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        shear_range=0.1,
        fill_mode='nearest'
    )
    
    # Validation data generator (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"✅ Number of classes: {train_generator.num_classes}")
    
    return train_generator, val_generator


def build_model(num_classes, img_size, learning_rate):
    """
    Build MobileNetV2 model with transfer learning
    
    Args:
        num_classes: Number of output classes
        img_size: Input image size
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    print(f"\n🏗️ Building model...")
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model built successfully")
    print(f"   Total parameters: {model.count_params():,}")
    
    return model


def train_model(model, train_gen, val_gen, epochs, model_dir):
    """
    Train the model with callbacks
    
    Args:
        model: Keras model to train
        train_gen: Training data generator
        val_gen: Validation data generator
        epochs: Number of epochs
        model_dir: Directory to save model
    
    Returns:
        Training history
    """
    print(f"\n🚀 Starting training for {epochs} epochs...")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    return history


def save_model(model, model_dir, model_name='traffic_sign_classifier.h5'):
    """
    Save the trained model
    
    Args:
        model: Trained Keras model
        model_dir: Directory to save model
        model_name: Model filename
    """
    model_path = os.path.join(model_dir, model_name)
    model.save(model_path)
    print(f"\n💾 Model saved to: {model_path}")


def evaluate_model(model, val_gen):
    """
    Evaluate model on validation set
    
    Args:
        model: Trained Keras model
        val_gen: Validation data generator
    """
    print(f"\n📊 Evaluating model on validation set...")
    
    val_loss, val_accuracy = model.evaluate(val_gen, verbose=0)
    
    print(f"\n{'='*50}")
    print(f"FINAL VALIDATION METRICS")
    print(f"{'='*50}")
    print(f"Validation Loss:     {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"{'='*50}\n")


def main():
    """Main training function"""
    # Parse arguments
    args = parse_arguments()
    
    print(f"\n{'='*50}")
    print(f"TRAFFIC SIGN CLASSIFICATION - TRAINING")
    print(f"{'='*50}")
    print(f"Configuration:")
    print(f"  Data directory:    {args.data_dir}")
    print(f"  Model directory:   {args.model_dir}")
    print(f"  Epochs:            {args.epochs}")
    print(f"  Batch size:        {args.batch_size}")
    print(f"  Image size:        {args.img_size}x{args.img_size}")
    print(f"  Validation split:  {args.val_split}")
    print(f"  Learning rate:     {args.learning_rate}")
    print(f"{'='*50}")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\n🖥️  GPU Available: {len(gpus) > 0}")
    if gpus:
        print(f"   GPU Device(s): {[gpu.name for gpu in gpus]}")
    
    # Create data generators
    train_gen, val_gen = create_data_generators(
        args.data_dir,
        args.img_size,
        args.batch_size,
        args.val_split
    )
    
    # Build model
    model = build_model(
        num_classes=train_gen.num_classes,
        img_size=args.img_size,
        learning_rate=args.learning_rate
    )
    
    # Train model
    history = train_model(
        model,
        train_gen,
        val_gen,
        args.epochs,
        args.model_dir
    )
    
    # Evaluate model
    evaluate_model(model, val_gen)
    
    # Save final model
    save_model(model, args.model_dir)
    
    print(f"\n✅ Training completed successfully!")
    print(f"📁 Model saved in: {args.model_dir}/")


if __name__ == '__main__':
    main()