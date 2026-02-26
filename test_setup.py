#!/usr/bin/env python3
"""
Test script to verify project setup and code structure
"""

import os
import sys

def test_project_structure():
    """Test if the project has the correct structure"""
    print("=== Testing Project Structure ===")
    
    # Check required directories
    required_dirs = ['Bacterial leaf blight', 'Brown spot', 'Leaf smut', 'Prediction_images']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name} directory found")
            # Count images in directory
            if os.path.isdir(dir_name):
                images = [f for f in os.listdir(dir_name) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.JPG'))]
                print(f"  - {len(images)} images found")
        else:
            print(f"✗ {dir_name} directory missing")
            missing_dirs.append(dir_name)
    
    # Check required files
    required_files = [
        'rice_disease_model.py',
        'train_model.py', 
        'streamlit_app.py',
        'requirements.txt',
        'README.md'
    ]
    
    print("\n=== Checking Required Files ===")
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✓ {file_name} found")
        else:
            print(f"✗ {file_name} missing")
    
    # Check Python version
    print(f"\n=== Python Environment ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Check if we can import basic modules
    print(f"\n=== Testing Basic Imports ===")
    try:
        import numpy as np
        print("✓ numpy available")
    except ImportError:
        print("✗ numpy not available")
    
    try:
        import pandas as pd
        print("✓ pandas available")
    except ImportError:
        print("✗ pandas not available")
    
    try:
        import PIL
        print("✓ PIL/Pillow available")
    except ImportError:
        print("✗ PIL/Pillow not available")
    
    try:
        import tensorflow as tf
        print("✓ tensorflow available")
    except ImportError:
        print("✗ tensorflow not available (will be installed during setup)")
    
    try:
        import streamlit as st
        print("✓ streamlit available")
    except ImportError:
        print("✗ streamlit not available (will be installed during setup)")
    
    return len(missing_dirs) == 0

def test_dataset_integrity():
    """Test dataset integrity"""
    print("\n=== Testing Dataset Integrity ===")
    
    total_images = 0
    for class_name in ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']:
        if os.path.exists(class_name):
            images = [f for f in os.listdir(class_name) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.JPG'))]
            total_images += len(images)
            print(f"{class_name}: {len(images)} images")
    
    print(f"Total images: {total_images}")
    
    if total_images >= 100:
        print("✓ Sufficient dataset size for training")
        return True
    else:
        print("✗ Dataset may be too small for effective training")
        return False

def main():
    """Main test function"""
    print("🌾 Rice Leaf Disease Prediction - Setup Test")
    print("=" * 50)
    
    # Test project structure
    structure_ok = test_project_structure()
    
    # Test dataset integrity
    dataset_ok = test_dataset_integrity()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    
    if structure_ok and dataset_ok:
        print("✓ Project structure is correct")
        print("✓ Dataset appears sufficient")
        print("\n🎉 Ready for setup! Follow the WINDOWS_SETUP_GUIDE.md")
    else:
        print("✗ Some issues found. Please check the output above.")
        
        if not structure_ok:
            print("- Missing directories or files")
        if not dataset_ok:
            print("- Dataset may be insufficient")
    
    print("\nNext steps:")
    print("1. Install Python 3.8-3.10")
    print("2. Create virtual environment")
    print("3. Install requirements: pip install -r requirements.txt")
    print("4. Run training: python train_model.py")
    print("5. Start app: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main() 