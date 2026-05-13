import torch, torchvision, torchaudio
print(f"PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
print(f"TorchVision: {torchvision.__version__}")
print(f"TorchAudio: {torchaudio.__version__}")

import transformers, accelerate, peft, bitsandbytes
print(f"Transformers: {transformers.__version__}")
print(f"Accelerate: {accelerate.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"Bitsandbytes: {bitsandbytes.__version__}")