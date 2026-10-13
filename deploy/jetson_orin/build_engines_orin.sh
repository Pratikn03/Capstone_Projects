#!/usr/bin/env bash
# Execute directly on NVIDIA Jetson AGX / Orin Nano with JetPack 5.x/6.x
set -e
echo "=========================================================="
echo "  COMPILING TENSORRT ENGINES ON JETSON ORIN (ARM64)"
echo "=========================================================="

echo "[1/2] Compiling Universal Physical AI Shield..."
trtexec --onnx=../../artifacts/models/physical_ai_shield_universal.onnx \
        --saveEngine=../../artifacts/models/physical_ai_shield_universal_orin.engine \
        --fp16 --dumpProfile

echo "[2/2] Compiling 3D World Model..."
trtexec --onnx=../../artifacts/models/world_model_3d_A100.onnx \
        --saveEngine=../../artifacts/models/world_model_3d_orin.engine \
        --fp16 --dumpProfile

echo "Engines compiled successfully for Orin Tensor Cores!"
