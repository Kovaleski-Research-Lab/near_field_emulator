import torch
import torch.nn.functional as F
from .WavePropModel import WavePropModel
from diffusers import DiffusionPipeline, UNet3DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0

class WaveDiffusion(WavePropModel):
    """
    A model leveraging a pretrained Stable Diffusion Image-to-Video (SVD) pipeline
    for generating future frames of wavefront propagation.

    This class uses the underlying UNet from the diffusion model for training,
    while maintaining the full pipeline for inference.
    """
    def __init__(self, model_config, fold_idx=None):
        super().__init__(model_config, fold_idx)
        self.target_size = 224  # CLIP's expected input size

    def create_architecture(self):
        # Load the full pipeline for inference
        self.pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", 
            use_safetensors=True
        ).to(self._device)
        
        # Extract the UNet for training
        self.unet = self.pipeline.unet
        
        # Configure model parameters
        self.num_generated_frames = self.conf.diffusion.num_generated_frames
        self.prompt = self.conf.diffusion.prompt
        self.use_half_precision = self.conf.diffusion.use_half_precision
        
        # Set up the UNet for training
        self.unet.train()
        self.unet.enable_gradient_checkpointing()
        
        # Use efficient attention
        self.unet.set_attn_processor(AttnProcessor2_0())
        
        # Freeze other components of the pipeline
        # Freeze VAE
        for param in self.pipeline.vae.parameters():
            param.requires_grad = False
            
        # Freeze image encoder
        for param in self.pipeline.image_encoder.parameters():
            param.requires_grad = False
            
        # Freeze scheduler (though it doesn't have parameters)
        #self.pipeline.scheduler.eval()
        
        # Freeze feature extractor (though it doesn't have parameters)
        #self.pipeline.feature_extractor.eval()

    def forward(self, x, meta=None):
        """
        Forward pass using the UNet for training.
        
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch, seq_len, r_i, xdim, ydim)
        meta: Optional[Any]
            Additional metadata, not used in this implementation
            
        Returns
        -------
        preds: torch.Tensor
            Predicted frames of shape (batch, seq_len, r_i, xdim, ydim)
        meta: None
            Not used in this implementation
        """
        batch, input_seq_len, r_i, height, width = x.size()
        
        if input_seq_len != 1:
            raise ValueError("WaveDiffusion currently supports only a single initial frame as input.")
        
        # Convert input to latent space
        initial_frame = x[:, 0]  # (batch, r_i, H, W)
        
        # Normalize and prepare input
        pseudo_rgb = torch.zeros(batch, 3, height, width, device=x.device)
        pseudo_rgb[:, 0] = (initial_frame[:, 0] - initial_frame[:, 0].min()) / (initial_frame[:, 0].max() - initial_frame[:, 0].min() + 1e-8)
        pseudo_rgb[:, 1] = (initial_frame[:, 1] - initial_frame[:, 1].min()) / (initial_frame[:, 1].max() - initial_frame[:, 1].min() + 1e-8)
        
        # Resize to CLIP's expected input size
        resized_rgb = F.interpolate(
            pseudo_rgb,
            size=(self.target_size, self.target_size),
            mode='bilinear',
            align_corners=False
        )
        
        # Get image embeddings
        image_embeddings = self.pipeline.image_encoder(resized_rgb)
        
        # Encode to latent space
        latents = self.pipeline.vae.encode(pseudo_rgb).latent_dist.sample()
        latents = latents * self.pipeline.vae.config.scaling_factor
        
        # Prepare noise schedule
        noise_scheduler = self.pipeline.scheduler
        noise = torch.randn_like(latents)
        
        # Initialize scheduler timesteps
        num_inference_steps = 50  # Standard number of inference steps
        noise_scheduler.set_timesteps(num_inference_steps, device=latents.device)
        
        # Sample timesteps from the initialized schedule
        timesteps = noise_scheduler.timesteps[torch.randint(
            0, 
            len(noise_scheduler.timesteps), 
            (batch,), 
            device=latents.device
        )]
        
        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Prepare added_time_ids for the UNet
        # Create base time ids with proper dimensions
        base_time_ids = torch.tensor([
            height, width,  # original size
            0, 0,          # crop top left
            height, width,  # target size
        ], device=latents.device, dtype=latents.dtype)
        
        # Reshape to match UNet's expected input
        base_time_ids = base_time_ids.unsqueeze(0)  # (1, 6)
        
        # Project to the correct dimension
        time_embeds = self.unet.add_embedding.linear_1(base_time_ids)  # (1, 1280)
        
        # Add sequence dimension and repeat for batch size
        time_embeds = time_embeds.unsqueeze(1)  # (1, 1, 1280)
        added_time_ids = time_embeds.repeat(batch, 1, 1)  # (batch, 1, 1280)
        
        # UNet forward pass
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
        ).sample
        
        # Decode predictions
        pred_frames = self.pipeline.vae.decode(model_pred / self.pipeline.vae.config.scaling_factor).sample
        
        # Convert back to real/imag format
        preds_real = pred_frames[:, 0]  # (batch, H, W)
        preds_imag = pred_frames[:, 1]  # (batch, H, W)
        
        # Stack and reshape to match expected output format
        preds = torch.stack([preds_real, preds_imag], dim=1)  # (batch, 2, H, W)
        preds = preds.unsqueeze(1)  # Add seq_len dimension: (batch, 1, 2, H, W)
        
        return preds, None

    def shared_step(self, batch, batch_idx):
        """
        Training step for the diffusion model.
        
        Parameters
        ----------
        batch: Tuple[torch.Tensor, torch.Tensor]
            Input batch containing samples and labels
        batch_idx: int
            Index of the current batch
            
        Returns
        -------
        loss: torch.Tensor
            Computed loss value
        preds: torch.Tensor
            Model predictions
        """
        samples, labels = batch
        preds, _ = self.forward(samples)
        loss_dict = self.objective(preds, labels)
        loss = loss_dict['loss']
        return loss, preds
    
    def organize_testing(self, preds, batch, batch_idx, dataloader_idx=0):
        """
        Organize testing results.
        
        Parameters
        ----------
        preds: torch.Tensor
            Model predictions
        batch: Tuple[torch.Tensor, torch.Tensor]
            Input batch
        batch_idx: int
            Batch index
        dataloader_idx: int
            Dataloader index (0 for validation, 1 for training)
        """
        samples, labels = batch
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        mode = 'valid' if dataloader_idx == 0 else 'train'
        self.test_results[mode]['nf_pred'].append(preds_np)
        self.test_results[mode]['nf_truth'].append(labels_np)