#!/usr/bin/env python3
"""
Test script to verify bias_decay parameter works correctly.
This script demonstrates how to use the new bias_decay parameter.
"""

from ultralytics import YOLO

# Example 1: Train with bias decay using command-line style args
print("=" * 80)
print("Testing bias_decay parameter")
print("=" * 80)

# Load a model
model = YOLO("yolo11n-cls.pt")

# Train with bias decay
# Note: Using epochs=1 and imgsz=32 for quick testing
print("\nTraining with bias_decay=0.0001...")
results = model.train(
    data="imagenet10",
    epochs=1,
    imgsz=32,
    batch=8,
    weight_decay=0.0005,  # Standard weight decay
    bias_decay=0.0001,     # NEW: Bias decay parameter
    verbose=True,
)

print("\n" + "=" * 80)
print("Test completed! Check the optimizer initialization message above.")
print("It should show: bias(decay=0.0001)")
print("=" * 80)
