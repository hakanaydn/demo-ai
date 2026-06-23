#!/bin/bash
set -e

echo "=== 1. Training model with TensorFlow (Python) ==="
python3 train.py

echo ""
echo "=== 2. Building C++ application ==="
mkdir -p build
cmake -B build -S .
cmake --build build

echo ""
echo "=== 3. Running C++ inference ==="
./build/add_app
