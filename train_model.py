#!/usr/bin/env python3
"""
Rice Leaf Disease Prediction Model Training Script
This script trains the CNN model for rice leaf disease classification.
"""

import os
import sys
from rice_disease_model import RiceDiseasePredictor
import tensorflow as tf

def main():
    print("=== Rice Leaf Disease Prediction Model Training ===")
    
    # Initialize the predictor
    predictor = RiceDiseasePredictor()
    
    # Check if dataset directories exist
    required_dirs = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"Error: Missing dataset directories: {missing_dirs}")
        print("Please ensure the dataset folders are present in the current directory.")
        return
    
    print("✓ Dataset directories found")
    
    # Setup data directories
    print("\n1. Setting up data directories...")
    train_test_path = predictor.setup_data_directories()
    print("✓ Data directories created")
    
    # Split dataset
    print("\n2. Splitting dataset into train/test sets...")
    predictor.split_dataset(train_test_path)
    print("✓ Dataset split completed (80% train, 20% test)")
    
    # Create data generators
    print("\n3. Creating data generators...")
    predictor.create_data_generators(train_test_path)
    print("✓ Data generators created")
    
    # Create and train Xception model (best performing model from the notebook)
    print("\n4. Creating Xception transfer learning model...")
    model = predictor.create_xception_model()
    print("✓ Xception model created")
    
    # Train model
    print("\n5. Training model...")
    print("Training with augmented data for better generalization...")
    history = predictor.train_model(model, use_augmentation=True, epochs=20)
    print("✓ Model training completed")
    
    # Save the model
    print("\n6. Saving trained model...")
    model_path = "rice_disease_model.h5"
    predictor.save_model(model, model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Evaluate model
    print("\n7. Evaluating model...")
    test_loss, test_accuracy = model.evaluate(predictor.test_data, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    print("\n=== Training Complete ===")
    print("The model is ready for use in the Streamlit application!")
    
    

if __name__ == "__main__":
    main() 