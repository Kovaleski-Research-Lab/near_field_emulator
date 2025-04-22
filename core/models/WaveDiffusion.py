import torch
from .WavePropModel import WavePropModel
from diffusers import DiffusionPipeline

class WaveDiffusion(WavePropModel):
    """
    A model leveraging a pretrained Stable Diffusion Image-to-Video (SVD) pipeline
    for generating future frames of wavefront propagation.

    This class assumes:
    - We have a pretrained SVD pipeline accessible from Hugging Face.
    - The pipeline: given a single image, generates a fixed number of subsequent frames.
    - We'll adapt the pipeline to accept a single "initial frame" from our dataset
      and produce a sequence of frames. We then align these predicted frames with 
      the ground-truth wave propagation frames for training.
    """

    def __init__(self, model_config, fold_idx=None):
        super().__init__(model_config, fold_idx)

    def create_architecture(self):
        # Load a pretrained image-to-video pipeline from Stability AI
        self.pipeline = DiffusionPipeline.from_pretrained(
             "stabilityai/stable-video-diffusion-img2vid", use_safetensors=True
        ).to(self._device)
        
        # Pseudocode (you'll need to replace with actual code from the SVD model):
        self.num_generated_frames = self.conf.diffusion.num_generated_frames
        self.prompt = self.conf.diffusion.prompt
        self.use_half_precision = self.conf.diffusion.use_half_precision
        
        #
        # If half precision is desired:
        # if self.use_half_precision:
        #     self.pipeline = self.pipeline.to(torch.float16)
        
        for param in self.pipeline.parameters():
            param.requires_grad = True

    def forward(self, x, meta=None):
        """
        Forward pass:
        x: (batch, seq_len, r_i, xdim, ydim) where typically seq_len=1 for the first input frame
           or if you do 'one_to_many', seq_len could be just 1 input frame.
        We'll:
          1. Extract the initial frame from x.
          2. Convert it into a suitable image format (e.g. pseudo-RGB).
          3. Pass it through the SVD pipeline to generate self.num_generated_frames.
          4. Convert generated frames back into the format (batch, seq_len, r_i, xdim, ydim).
        """
        
        # For simplicity, assume one-to-many mode: 
        # One input frame -> Multiple output frames
        batch, input_seq_len, r_i, height, width = x.size()
        if input_seq_len != 1:
            raise ValueError("WaveSVDModel currently supports only a single initial frame as input.")
        
        # Convert the complex field input into an image format that SVD can handle.
        # Example: real & imag channels => 2-channel. SVD expects 3-channel (RGB), 
        # so we can replicate one channel or create a dummy third channel.
        
        initial_frame = x[:, 0]  # shape: (batch, r_i, H, W), where r_i=2 typically (real/imag)
        
        # Normalize to [0,1] and create a pseudo-RGB image: 
        # E.g., (r_i=2) -> (R=real, G=imag, B=zeros)
        # Ensure no negative values and scale appropriately if needed.
        # This is just a placeholder normalization; adapt as needed.
        pseudo_rgb = torch.zeros(batch, 3, height, width, _device=x._device)
        pseudo_rgb[:, 0] = (initial_frame[:, 0] - initial_frame[:, 0].min()) / (initial_frame[:, 0].max() - initial_frame[:, 0].min() + 1e-8)
        pseudo_rgb[:, 1] = (initial_frame[:, 1] - initial_frame[:, 1].min()) / (initial_frame[:, 1].max() - initial_frame[:, 1].min() + 1e-8)
        pseudo_rgb[:, 2] = torch.zeros_like(pseudo_rgb[:, 1])
        
        # NOTE: This code assumes batch=1 for simplicity. For larger batch sizes, 
        # you'd likely run one sample at a time or modify the pipeline for batch processing.
        if batch > 1:
            raise NotImplementedError("For simplicity, this example only handles batch=1 with SVD.")
        
        # format input image for pipeline
        input_img = pseudo_rgb.squeeze(0).permute(1,2,0) # (H,W,3)
        
        # Run through the SVDiffusion pipeline:
        results = self.pipeline(prompt=self.prompt, image=input_img, num_frames=self.num_generated_frames)
        predicted_frames = results.frames
        
        # convert to tensor
        predicted_frames = torch.tensor(predicted_frames).unsqueeze(0) # (1, seq_len, 3, H, W)
        
        # Convert frames back to real/imag format. We had R=real, G=imag. We'll ignore B or just trust it matches G.
        # Inverse normalization (this is just conceptual; you should store normalization stats and invert properly).
        # Let's just say we take the first two channels as real/imag and skip the third for now.
        preds_real = predicted_frames[:, :, 0]  # shape: (1, seq_len, H, W)
        preds_imag = predicted_frames[:, :, 1]  # shape: (1, seq_len, H, W)
        
        # Combine back into (batch, seq_len, r_i=2, H, W)
        preds = torch.stack([preds_real, preds_imag], dim=2)  # (1, seq_len, 2, H, W)
        
        return preds, None  # meta not used here

    def shared_step(self, batch, batch_idx):
        samples, labels = batch
        # labels shape: (batch, seq_len, r_i, xdim, ydim)
        # samples shape: (batch, input_seq_len, r_i, xdim, ydim)
        
        preds, _ = self.forward(samples)
        
        # Compute loss using base objective
        loss_dict = self.objective(preds, labels)
        loss = loss_dict['loss']

        return loss, preds
    
    def organize_testing(self, preds, batch, batch_idx, dataloader_idx=0):
        samples, labels = batch
        preds_np = preds.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        # Determine the mode based on dataloader_idx
        if dataloader_idx == 0:
            mode = 'valid'
        elif dataloader_idx == 1:
            mode = 'train'
        else:
            raise ValueError(f"Invalid dataloader index: {dataloader_idx}")
        
        self.test_results[mode]['nf_pred'].append(preds_np)
        self.test_results[mode]['nf_truth'].append(labels_np)