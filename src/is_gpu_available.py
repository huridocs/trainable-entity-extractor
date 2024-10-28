import torch

if __name__ == "__main__":
    print("GPU", torch.cuda.is_available())
