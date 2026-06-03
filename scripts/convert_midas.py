#!/usr/bin/env python3
"""
Sync Vision — MiDaS Depth Model Conversion Script
===================================================
Converts the MiDaS v2.1 Small PyTorch model to TFLite format.

Steps:
  1. Download the PyTorch model:
     wget https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.pt
  
  2. Run this conversion script:
     python3 convert_midas.py --input model-small.pt --output /path/to/midas_v21_small_256.tflite

Requirements:
  pip install torch torchvision tensorflow

Alternative: Use MediaPipe Depth Estimation model instead:
  Kaggle: https://www.kaggle.com/models/mediapipe/depth-estimation
  This gives a ready-to-use TFLite depth model.
"""

import os
import sys
import argparse
import numpy as np


def convert_midas_to_tflite(input_path, output_path):
    """Convert MiDaS PyTorch model to TFLite via ONNX."""
    try:
        import torch
        import tensorflow as tf
    except ImportError:
        print("ERROR: PyTorch and TensorFlow required.")
        print("Install with: pip install torch tensorflow")
        sys.exit(1)
    
    INPUT_SIZE = 256
    
    print(f"Loading MiDaS model from: {input_path}")
    
    # Load PyTorch model
    device = torch.device('cpu')
    model = torch.load(input_path, map_location=device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    
    # Export to ONNX first
    onnx_path = input_path.replace('.pt', '.onnx')
    print(f"Exporting to ONNX: {onnx_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=None
    )
    
    # Convert ONNX to TFLite
    print("Converting ONNX to TFLite...")
    
    # This requires onnx-tf package
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        
        # Save as SavedModel
        saved_model_path = onnx_path.replace('.onnx', '_saved_model')
        tf_rep.export_graph(saved_model_path)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size = os.path.getsize(output_path)
        print(f"\n✓ Model exported to: {output_path}")
        print(f"  Size: {model_size / 1024 / 1024:.1f} MB")
        
    except ImportError:
        print("ERROR: onnx-tf not installed.")
        print("Install with: pip install onnx-tf")
        print("")
        print("Alternative: Use the onnx2tf tool:")
        print("  pip install onnx2tf")
        print(f"  onnx2tf -i {onnx_path} -o {os.path.dirname(output_path)}")
        sys.exit(1)
    
    # Cleanup
    if os.path.exists(onnx_path):
        os.remove(onnx_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert MiDaS PyTorch model to TFLite')
    parser.add_argument('--input', required=True, help='Path to MiDaS PyTorch model (.pt)')
    parser.add_argument('--output', default=None, help='Output TFLite model path')
    
    args = parser.parse_args()
    
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        args.output = os.path.join(
            project_dir, 'app', 'src', 'main', 'assets', 'models', 'midas_v21_small_256.tflite'
        )
    
    convert_midas_to_tflite(args.input, args.output)
