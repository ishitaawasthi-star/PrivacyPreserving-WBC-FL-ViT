import sys
print("Python:", sys.version)
try:
    import torch, torchvision, streamlit
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("Streamlit:", streamlit.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Environment check passed.")
except Exception as exc:
    print("Environment check failed:", exc)
    raise
