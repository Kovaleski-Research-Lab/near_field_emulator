import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
from skimage.metrics import structural_similarity as ssim
import cv2
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

class Encoder(nn.Module):
    """Encodes E to a representation space for input to RNN"""
    def __init__(self, channels, conf):
        super().__init__()
        
        self.method = conf.method
        self.latent_dim = conf.latent_dim
        self.spatial = conf.spatial
        self.layers = nn.ModuleList()
        
        if self.method == 'conv':
            # Calculate number of downsampling steps needed
            current_size = self.spatial
            target_size = self.latent_dim
            num_downsamples = int(np.log2(current_size / target_size))
            
            # Build encoder layers dynamically based on config
            layers = []
            in_channels = 2  # Start with 2 channels (real/imag)
            
            for i in range(num_downsamples):
                out_channels = channels[i] if i < len(channels) else channels[-1]
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                ])
                in_channels = out_channels
            
            # Final conv to maintain size
            layers.extend([
                nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(2),
                nn.LeakyReLU(0.2)
            ])
            
            self.layers = nn.Sequential(*layers)
            self.final_size = target_size
            self.final_channels = 2
            
        else:
            raise ValueError(f"Unsupported encoding method: {self.method}")
            
    def forward(self, x):
        # x shape [batch, channels, xdim, ydim]
        x = self.layers(x)
        return x
    
class Decoder(nn.Module):
    """Decodes latent representation back to E"""
    def __init__(self, channels, conf):
        super().__init__()
        
        self.channels = channels
        self.method = conf.method
        self.latent_dim = conf.latent_dim
        self.spatial = conf.spatial
        
        if self.method == 'conv':
            # Calculate number of upsampling steps needed
            current_size = self.latent_dim
            target_size = self.spatial
            num_upsamples = int(np.log2(target_size / current_size))
            
            # Build decoder layers dynamically based on config
            layers = []
            in_channels = 2  # Start with 2 channels (real/imag)
            
            for i in range(num_upsamples):
                out_channels = channels[i] if i < len(channels) else channels[-1]
                layers.extend([
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2)
                ])
                in_channels = out_channels
            
            # Final conv to maintain size
            layers.extend([
                nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(2),
                nn.Tanh()  # Use tanh for final activation to bound output
            ])
            
            self.layers = nn.Sequential(*layers)
            
    def forward(self, x):
        # x shape: [batch, 2, latent_dim, latent_dim]
        x = self.layers(x)
        # Ensure exact size match
        if x.shape[-1] != self.spatial or x.shape[-2] != self.spatial:
            x = x[:, :, :self.spatial, :self.spatial]
        return x  # [batch, 2, spatial, spatial]
    
class Autoencoder(LightningModule):
    def __init__(self, model_config, fold_idx=None):
        super().__init__()
        
        self.conf = model_config
        self.encoder_channels = self.conf.autoencoder.encoder_channels
        self.decoder_channels = self.conf.autoencoder.decoder_channels
        self.spatial_size = self.conf.autoencoder.spatial
        self.encoder = Encoder(self.encoder_channels, conf=self.conf.autoencoder)
        self.decoder = Decoder(self.decoder_channels, conf=self.conf.autoencoder)
        self.learning_rate = self.conf.learning_rate
        self.lr_scheduler = self.conf.lr_scheduler
        self.loss_func = self.conf.objective_function
        self.fold_idx = fold_idx
        
        # Create directory for latent space visualizations
        self.latent_viz_dir = os.path.join(f'/develop/results/meep_meep/refractive_idx/autoencoder/model_{self.conf.model_id}/', 'latent_viz')
        os.makedirs(self.latent_viz_dir, exist_ok=True)
                
        self.test_results = {'train': {'nf_pred': [], 'nf_truth': []},
                             'valid': {'nf_pred': [], 'nf_truth': []}}
        
        # Initialize metrics
        self.train_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.train_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        
        self.save_hyperparameters()
        
    def visualize_latent_space(self, z):
        """
        Visualize the first channel of the latent space as a heatmap.
        
        Parameters
        ----------
        z: torch.Tensor
            Latent space tensor of shape [batch, 2, 14, 14]
        """
        # Get the first channel of the first batch item
        latent_viz = z[0, 0].detach().cpu().numpy()
        
        # Create figure
        plt.figure(figsize=(8, 8))
        plt.imshow(latent_viz, cmap='viridis')
        plt.colorbar()
        plt.title('Latent Space Visualization (First Channel)')
        
        # Save to file (overwrite existing)
        save_path = os.path.join(self.latent_viz_dir, 'latent_space.png')
        plt.savefig(save_path)
        plt.close()
        
    def state_dict(self, *args, **kwargs):
        """override state dict to organize things better"""
        state_dict = super().state_dict(*args, **kwargs)
        
        # create nested state dict to separate encoder and decoder
        nested_state_dict = {}
        
        # organize
        for key, value in state_dict.items():
            if key.startswith('encoder'):
                nested_state_dict[key] = value
            elif key.startswith('decoder'):
                nested_state_dict[key] = value
            
        return nested_state_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle both nested and flat structures  
           Handles older pretrained models that weren't organized with above override"""
        # Check if we have a nested or flat structure
        if 'encoder' in state_dict and 'decoder' in state_dict:
            # Nested structure - flatten it
            flat_state_dict = {}
            for module in ['encoder', 'decoder']:
                for key, value in state_dict[module].items():
                    flat_state_dict[f'{module}.{key}'] = value
            state_dict = flat_state_dict
            
        return super().load_state_dict(state_dict, strict=strict)
    
    def compute_loss(self, preds, labels, choice):
        """
        Compute loss given predictions and labels.
        """
        if choice == 'mse':
            # Mean Squared Error
            preds = preds.to(torch.float32).contiguous()
            labels = labels.to(torch.float32).contiguous()
            fn = torch.nn.MSELoss()
            loss = fn(preds, labels)
        elif choice == 'emd':
            # ignoring emd for now
            raise NotImplementedError("Earth Mover's Distance not implemented!")
        elif choice == 'psnr':
            # Peak Signal-to-Noise Ratio
            preds, labels = preds.unsqueeze(1), labels.unsqueeze(1)  # channel dim
            fn = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
            psnr_value = fn(preds, labels)
            loss = (1 - psnr_value)
        elif choice == 'ssim':
            # Structural Similarity Index
            if preds.size(-1) < 11 or preds.size(-2) < 11:
                loss = 0 # if the size is too small, SSIM is not defined
            else:
                torch.use_deterministic_algorithms(True, warn_only=True)
                with torch.backends.cudnn.flags(enabled=False):
                    fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
                    ssim_value = fn(preds, labels)
                    ssim_comp = (1 - ssim_value)
                #loss = ssim_comp
                # Mean Squared Error
                preds = preds.to(torch.float32).contiguous()
                labels = labels.to(torch.float32).contiguous()
                fn2 = torch.nn.MSELoss()
                mse_comp = fn2(preds, labels)
                loss = self.conf.mcl_params['alpha'] * mse_comp + self.conf.mcl_params['beta'] * ssim_comp
        else:
            raise ValueError(f"Unsupported loss function: {choice}")
            
        return loss
        
    def forward(self, x):
        # [batch, 2, xdim, ydim] -> x
        z = self.encoder(x)
        # Visualize latent space
        self.visualize_latent_space(z)
        x_hat = self.decoder(z)
        return x_hat
    
    def shared_step(self, batch, stage="train"):
        x, m = batch
        if len(x.shape) == 5:
            x = x[:, 0] # [batch, 2, xdim, ydim]
            
        x_hat = self(x)
        #loss = F.mse_loss(x_hat, x)
        loss = self.compute_loss(x_hat, x, choice=self.loss_func)
        
        # calculate addl metrics
        with torch.no_grad():
            psnr = self.train_psnr(x_hat, x) if stage == 'train' else self.val_psnr(x_hat, x)
            ssim = self.train_ssim(x_hat, x) if stage == 'train' else self.val_ssim(x_hat, x)
            
        return {
            'loss': loss,
            'psnr': psnr,
            'ssim': ssim,
            'recon': x_hat
        }

    
    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch, 'train')
        
        # Log all metrics
        self.log('train_loss', results['loss'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_psnr', results['psnr'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_ssim', results['ssim'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        return results
    
    def validation_step(self, batch, batch_idx):
        results = self.shared_step(batch, 'valid')
        
        # Log all metrics
        self.log('val_loss', results['loss'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_psnr', results['psnr'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_ssim', results['ssim'], prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        
        return results
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # LR scheduler setup
        if self.lr_scheduler == 'CosineAnnealingLR':
            lr_scheduler = CosineAnnealingLR(optimizer, 
                                             T_max=100 * 0.25,
                                             eta_min=1e-6)
        elif self.lr_scheduler == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, 
                                             mode='min', 
                                             factor=0.5, 
                                             patience=3, 
                                             min_lr=1e-6, 
                                             threshold=0.001, 
                                             cooldown=2
                                            )
        else:
            raise ValueError(f"Unsupported LR scheduler: {self.lr_scheduler}")
        lr_scheduler_config = {"scheduler": lr_scheduler,
                               "interval": "epoch", "frequency": 1,
                               "monitor": "val_loss"}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        results = self.shared_step(batch, 'test')
        self.organize_testing(results['recon'], batch, batch_idx, dataloader_idx)
        
    def organize_testing(self, preds, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        preds = preds.detach().cpu().numpy()
        labels = x.detach().cpu().numpy()
        
        # Determine the mode based on dataloader_idx
        if dataloader_idx == 0:
            mode = 'valid'
        elif dataloader_idx == 1:
            mode = 'train'
        else:
            raise ValueError(f"Invalid dataloader index: {dataloader_idx}")
        
        # Append predictions and truths
        self.test_results[mode]['nf_pred'].append(preds)
        self.test_results[mode]['nf_truth'].append(labels)

    def on_train_epoch_end(self):
        # Get current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True)

    def on_test_end(self):
        for mode in ['train', 'valid']:
            if self.test_results[mode]['nf_pred']:
                self.test_results[mode]['nf_pred'] = np.concatenate(self.test_results[mode]['nf_pred'], axis=0)
                self.test_results[mode]['nf_truth'] = np.concatenate(self.test_results[mode]['nf_truth'], axis=0)
            else:
                print(f"No test results for mode: {mode}")