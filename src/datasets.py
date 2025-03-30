import os
from torch.utils.data import Dataset
from PIL import Image

class Vimeo90KSingleFrameDataset(Dataset):
    """Vimeo90K dataset loader for single frames"""
    def __init__(self, root_dir, transform=None, split='train'):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.meta_info = self._load_metadata()
        
    def _load_metadata(self):
        split_file = 'sep_testlist.txt' if self.split == 'test' else 'sep_trainlist.txt'
        with open(os.path.join(self.root_dir, split_file), 'r') as f:
            return [line.strip() for line in f if line.strip()]
        
    def __len__(self):
        return len(self.meta_info)
    
    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, "sequences", self.meta_info[idx], "im4.png")
        img = Image.open(path).convert('RGB')
        return self.transform(img) if self.transform else img

class KodakDataset(Dataset):
    """Kodak dataset loader for evaluation"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.png')])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        return self.transform(img) if self.transform else img