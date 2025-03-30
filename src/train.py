import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .datasets import Vimeo90KSingleFrameDataset
from .utils import (load_config, AverageMeter,  compute_psnr, init_lpips, create_train_transform)
from compressai.zoo import image_models
from compressai.losses import RateDistortionLoss

class Trainer:
    def __init__(self):
        self.config = load_config("train")
        self.device = torch.device("cuda" if self.config["use_cuda"] and torch.cuda.is_available() else "cpu")
        self.model = self._init_model()
        self.optimizer, self.aux_optimizer = self._init_optimizers()
        self.criterion = RateDistortionLoss(lmbda=self.config["lmbda"])
        self.lpips = init_lpips(self.device)
        self.train_loader = self._init_dataloader()
        self.best_loss = float('inf')
        
    def _init_model(self):
        model = image_models[self.config["model"]](quality=2)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        return model.to(self.device)
    
    def _init_optimizers(self):
        params = []
        aux_params = []
        for name, param in self.model.named_parameters():
            if "aux" in name:
                aux_params.append(param)
            else:
                params.append(param)
        return (
            torch.optim.Adam(params, lr=self.config["learning_rate"]),
            torch.optim.Adam(aux_params, lr=self.config["aux_learning_rate"])
        )
    
    def _init_dataloader(self):
        transform = create_train_transform()
        dataset = Vimeo90KSingleFrameDataset(
            self.config["dataset_path"],
            transform=transform,
            split='train'
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
            pin_memory=(self.device.type == "cuda")
        )
    
    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        psnrs = AverageMeter()
        bpps = AverageMeter()
        lpips_vals = AverageMeter()
        
        for images, _ in self.train_loader:
            images = images.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            self.aux_optimizer.zero_grad()
            out = self.model(images)
            
            # Calculate losses
            out_criterion = self.criterion(out, images)
            x_lpips = (images * 2) - 1
            x_hat_lpips = (out["x_hat"] * 2) - 1
            lpips_loss = self.lpips(x_lpips, x_hat_lpips).mean()
            
            # Combined loss
            loss = 0.1 * out_criterion["mse_loss"] + 0.9 * lpips_loss + 10 * self.criterion.lmbda * out_criterion["bpp_loss"]
            
            # Backward pass
            loss.backward()
            if self.config["clip_max_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["clip_max_norm"])
            self.optimizer.step()
            self.aux_optimizer.step()
            
            # Update metrics
            psnr = compute_psnr(images, out["x_hat"])
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            psnrs.update(psnr, batch_size)
            bpps.update(out_criterion["bpp_loss"].item(), batch_size)
            lpips_vals.update(lpips_loss.item(), batch_size)
            
        return losses.avg, psnrs.avg, bpps.avg, lpips_vals.avg
    
    def save_checkpoint(self, filename, is_best=False):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "aux_optimizer": self.aux_optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": self.best_loss,
        }
        path = os.path.join(self.config["checkpoint_dir"], filename)
        torch.save(state, path)
        if is_best:
            best_path = os.path.join(self.config["checkpoint_dir"], "best.pth")
            torch.save(state, best_path)
    
    def run(self):
        for epoch in range(self.config["epochs"]):
            loss, psnr, bpp, lpips_val = self.train_epoch(epoch)
            
            print(f"Epoch {epoch+1}/{self.config['epochs']}:")
            print(f"Loss: {loss:.4f} | PSNR: {psnr:.2f} dB")
            print(f"BPP: {bpp:.4f} | LPIPS: {lpips_val:.4f}")
            
            # Save checkpoints
            if (epoch + 1) % self.config["save_interval"] == 0:
                self.save_checkpoint(f"epoch_{epoch+1}.pth")
                
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_checkpoint("best.pth", is_best=True)
                print(f"New best checkpoint saved with loss {loss:.4f}")

def main():
    Trainer().run()

if __name__ == "__main__":
    main()