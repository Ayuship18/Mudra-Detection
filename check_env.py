"""
Quick sanity check that PyTorch can see your M2 GPU (MPS backend).
Run this first: python check_env.py
"""
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS built into this PyTorch install: {torch.backends.mps.is_built()}")
print(f"MPS available right now: {torch.backends.mps.is_available()}")

if torch.backends.mps.is_available():
    print("\n✅ You're good to go — training will use your M2 GPU (device='mps').")
else:
    print("\n⚠️  MPS not available. Training will fall back to CPU, which will be much")
    print("   slower. Make sure you're on macOS 12.3+ and have a recent PyTorch")
    print("   (pip install --upgrade torch torchvision).")