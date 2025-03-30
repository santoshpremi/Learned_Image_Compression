import torch
from torch.utils.data import DataLoader
from .datasets import KodakDataset
from .utils import (load_config, AverageMeter, 
                   compute_psnr, init_lpips,
                   create_eval_transform)
from compressai.zoo import image_models

class Evaluator:
    def __init__(self):
        self.config = load_config("eval")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.lpips = init_lpips(self.device)
        self.test_loader = self._init_dataloader()
        
    def _load_model(self):
        model = image_models[self.config["model"]](quality=2)
        state_dict = torch.load(self.config["checkpoint_path"], map_location=self.device)
        model.load_state_dict(state_dict["model"])
        return model.to(self.device).eval()
    
    def _init_dataloader(self):
        transform = create_eval_transform(self.config["patch_size"])
        dataset = KodakDataset(self.config["dataset_path"], transform=transform)
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            pin_memory=True
        )
    
    def evaluate(self):
        metrics = {
            'psnr': AverageMeter(),
            'bpp': AverageMeter(),
            'lpips': AverageMeter()
        }
        
        with torch.no_grad():
            for images, _ in self.test_loader:
                images = images.to(self.device)
                out = self.model(images)
                
                # Calculate metrics
                psnr = compute_psnr(images, out["x_hat"])
                bpp = out["bpp_loss"].mean().item()
                lpips_val = self.lpips(images, out["x_hat"]).mean().item()
                
                # Update meters
                batch_size = images.size(0)
                metrics['psnr'].update(psnr, batch_size)
                metrics['bpp'].update(bpp, batch_size)
                metrics['lpips'].update(lpips_val, batch_size)
                
        return metrics

def main():
    evaluator = Evaluator()
    metrics = evaluator.evaluate()
    print("\nEvaluation Results:")
    print(f"PSNR: {metrics['psnr'].avg:.2f} dB")
    print(f"BPP: {metrics['bpp'].avg:.4f}")
    print(f"LPIPS: {metrics['lpips'].avg:.4f}")

if __name__ == "__main__":
    main()