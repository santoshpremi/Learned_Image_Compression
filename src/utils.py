import yaml
import os
import math
import torch
import lpips
from pathlib import Path
from torchvision import transforms

class AverageMeter:
    """Computes and stores average values"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_config(config_name):
    """Load YAML config file with environment variable overrides"""
    config_path = Path(__file__).parent.parent / "config" / f"{config_name}_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Handle environment variable overrides
    config["dataset_path"] = os.getenv("DATASET_PATH", config.get("dataset_path", ""))
    config["checkpoint_path"] = os.getenv("CHECKPOINT_PATH", config.get("checkpoint_path", ""))
    
    # Auto-create checkpoint directory
    if "checkpoint_dir" in config:
        os.makedirs(config["checkpoint_dir"], exist_ok=True)
    
    return config

def compute_psnr(a, b, max_val=1.0):
    """Calculate PSNR between tensors"""
    mse = torch.mean((a - b) ** 2).item()
    return 20 * math.log10(max_val) - 10 * math.log10(mse)

def init_lpips(device):
    """Initialize LPIPS metric"""
    return lpips.LPIPS(net='alex').to(device)

def create_train_transform():
    """Training data transformations"""
    return transforms.Compose([
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])

def create_eval_transform(patch_size):
    """Evaluation data transformations"""
    return transforms.Compose([
        transforms.Resize(patch_size),
        transforms.ToTensor()
    ])