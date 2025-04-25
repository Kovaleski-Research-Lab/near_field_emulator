import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .WavePropModel import WavePropModel

# Helper function for padding
def get_padding_2d(input_shape, patch_size):
    """Calculate padding needed for height and width."""
    H, W = input_shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    # Pad evenly on both sides: (left, right, top, bottom)
    return (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

# ------------------------------------
# WaveTransformer Implementation
# ------------------------------------
class WaveTransformer(WavePropModel):
    """
    Near Field Wave Propagation Prediction Model using a Transformer architecture.

    Treats the input sequence of near-field maps like a video,
    using patch embeddings, positional encodings, and a Transformer
    Encoder-Decoder structure.
    """
    def __init__(self, model_config, fold_idx=None):
        # Define default transformer parameters before calling super().__init__
        # These can be overridden by the model_config
        self.arch_conf = model_config.transformer
        self.patch_size = self.arch_conf.patch_size
        self.embed_dim = self.arch_conf.embed_dim
        self.depth = self.arch_conf.depth # Total layers (e.g., 6 encoder, 6 decoder)
        self.num_heads = self.arch_conf.num_heads
        self.mlp_ratio = self.arch_conf.mlp_ratio
        self.dropout = model_config.dropout
        # input shape needs to be known for positional embeddings
        self.input_height = model_config.near_field_dim
        self.input_width = model_config.near_field_dim
        
        self.t_in_for_pos_embed = 1 # Define how many input steps pos embed should handle
        self.max_steps = self.t_in_for_pos_embed + model_config.seq_len
        
        super().__init__(model_config, fold_idx)
      
    def _create_upsample_block(self, in_channels, out_channels):
        """Helper function to create a ConvTranspose2d block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.GELU()
        )      
        
    def create_architecture(self):
        C = 2 # Input channels (real, imag)
        H, W = self.input_height, self.input_width
        padding_dims = get_padding_2d((H, W), self.patch_size)
        self.padded_H = H + padding_dims[2] + padding_dims[3]
        self.padded_W = W + padding_dims[0] + padding_dims[1]
        self.grid_size = (self.padded_H // self.patch_size, self.padded_W // self.patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        N = self.num_patches

        # --- Layers ---
        # 1. Patch Embedding (used for input and feedback)
        self.patch_embed = nn.Conv2d(C, self.embed_dim,
                                     kernel_size=self.patch_size,
                                     stride=self.patch_size)

        # 2. Positional Embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, N, self.embed_dim)) # Spatial
        self.temporal_embed = nn.Parameter(torch.zeros(1, self.max_steps, self.embed_dim)) # Temporal

        # 3. Transformer Block (Encoder Layers)
        transformer_layer = nn.TransformerEncoderLayer(
             d_model=self.embed_dim,
             nhead=self.num_heads,
             dim_feedforward=int(self.embed_dim * self.mlp_ratio),
             dropout=self.dropout,
             activation=F.gelu,
             batch_first=True # Crucial for easier handling
        )
        # Use TransformerEncoder with attention
        self.transformer_block = nn.TransformerEncoder(transformer_layer, num_layers=self.depth)
        # LayerNorm before prediction head is common
        self.norm = nn.LayerNorm(self.embed_dim)

        # 4. Prediction Head
        # Takes the final embedding (D) and predicts all patch features (N*P*P*C)
        # Needs to reconstruct the full spatial frame from a single vector per time step
        '''bottleneck_channels = self.embed_dim # e.g., 256
        up_channels = [bottleneck_channels, bottleneck_channels // 2, bottleneck_channels // 4, bottleneck_channels // 8] # e.g., [256, 128, 64, 32]
        if up_channels[-1] < C: # Ensure last channel count is at least C
             up_channels[-1] = C

        # Project D embedding to start the spatial grid (grid_H x grid_W)
        self.head_bottleneck_proj = nn.Linear(self.embed_dim, up_channels[0] * self.grid_size[0] * self.grid_size[1])

        # Upsampling blocks (4 stages to go from 11x11 -> 176x176)
        self.upsample_block1 = self._create_upsample_block(up_channels[0], up_channels[1]) # 11x11 -> 22x22
        self.upsample_block2 = self._create_upsample_block(up_channels[1], up_channels[2]) # 22x22 -> 44x44
        self.upsample_block3 = self._create_upsample_block(up_channels[2], up_channels[3]) # 44x44 -> 88x88
        # Final block goes to C=2 channels aannd no batchnorm needed prob
        self.upsample_block4 = nn.ConvTranspose2d(up_channels[3], C, kernel_size=4, stride=2, padding=1) # 88x88 -> 176x176'''
        
        # MLP PREDICTION HEAD
        self.prediction_head_mlp = nn.Sequential(
            # Map D -> N*D (Generate embeddings for all patch locations)
            nn.Linear(self.embed_dim, N * self.embed_dim),
            nn.GELU()
        )
        # Project each generated patch embedding D -> P*P*C
        self.patch_projection = nn.Linear(self.embed_dim, self.patch_size * self.patch_size * C)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.temporal_embed, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def _embed_frame(self, frame, t_step):
        """Embeds a single frame (B, C, H, W) at time step t_step."""
        B, C, H, W = frame.shape
        N = self.num_patches
        D = self.embed_dim

        padding = get_padding_2d((H, W), self.patch_size)
        frame_padded = F.pad(frame, padding) # (B, C, H_pad, W_pad)

        # Patch embed -> Flatten -> Permute
        patch_emb = self.patch_embed(frame_padded).flatten(2).permute(0, 2, 1) # (B, N, D)

        # Add spatial embedding
        patch_emb_spat = patch_emb + self.pos_embed # (B, N, D)

        # Add temporal embedding for the given step
        patch_emb_spat_temp = patch_emb_spat + self.temporal_embed[:, t_step, :].unsqueeze(1) # (B, N, D)

        # Reshape for sequence concatenation: (B, 1*N, D)
        return patch_emb_spat_temp.view(B, N, D).reshape(B, 1 * N, D)
    
    def _generate_square_subsequent_mask(self, sz, device):
        """Generates causal mask for sz x sz."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def _predict_frame_from_embedding(self, embedding):
        """Predicts a frame (B, C, H, W) from the Transformer output embedding (B, D)."""
        B, D = embedding.shape
        N = self.num_patches
        C = 2 # Assuming C=2
        H, W = self.input_height, self.input_width
        P = self.patch_size
        grid_H, grid_W = self.grid_size

        # 1. Project D -> N*D
        projected_patches = self.prediction_head_mlp(embedding) # (B, N*D)

        # 2. Reshape to (B, N, D)
        patch_embeddings = projected_patches.view(B, N, D)

        # 3. Project each patch D -> P*P*C
        # Input (B, N, D) -> Reshape (B*N, D) -> Linear -> (B*N, P*P*C)
        patch_pixels_flat = self.patch_projection(patch_embeddings.view(-1, D)) # (B*N, P*P*C)

        # 4. Unpatch: Reshape and permute to form image grid
        # -> (B, N, P*P*C)
        patch_pixels_flat = patch_pixels_flat.view(B, N, -1)
        # -> (B, grid_H, grid_W, P*P*C)
        patch_pixels_grid = patch_pixels_flat.view(B, grid_H, grid_W, P*P*C)
        # -> (B, grid_H, grid_W, P, P, C)
        patch_pixels_6d = patch_pixels_grid.view(B, grid_H, grid_W, P, P, C)
        # -> Permute (B, C, grid_H, P, grid_W, P)
        patch_pixels_perm = patch_pixels_6d.permute(0, 5, 1, 3, 2, 4)
        # -> Reshape (B, C, H_pad, W_pad)
        frame_padded = patch_pixels_perm.reshape(B, C, self.padded_H, self.padded_W)

        # 5. Crop padding
        padding = get_padding_2d((H, W), P)
        crop_h_start = padding[2]; crop_h_end = self.padded_H - padding[3]
        crop_w_start = padding[0]; crop_w_end = self.padded_W - padding[1]
        frame = frame_padded[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]

        return frame # (B, C, H, W) 
            
    def forward(self, x, labels=None, meta=None):
        """
        Autoregressive forward pass. Uses teacher forcing during training.

        Args:
            x (torch.Tensor): Input frames (B, T_in, C, H, W). Assumes T_in=1 for now.
            labels (torch.Tensor, optional): Ground truth labels (B, T_out, C, H, W).
                                            Used for teacher forcing during training.
        Returns:
            Tuple[torch.Tensor, None]: Predicted sequence (B, T_out, C, H, W), None
        """
        B, T_in, C, H, W = x.shape
        T_out = self.seq_len
        N = self.num_patches
        D = self.embed_dim
        device = x.device

        # Assert T_in == 1 for simplicity in this implementation
        if T_in != 1 and self.t_in_for_pos_embed != T_in:
             print(f"[Warning] T_in={T_in} but expecting {self.t_in_for_pos_embed}. Adjust config/model if needed.")
             # For now, proceed assuming T_in=1 for embedding indexing logic
             T_in = 1 # Override T_in for indexing consistency if mismatch

        if self.training and labels is not None:
            # --- Teacher Forcing during Training ---
            # 1. Prepare Input Sequence
            target_input_frames = torch.cat([x, labels[:, :-1, ...]], dim=1) # (B, T_out, C, H, W)
            T = target_input_frames.shape[1] # T = T_out (assuming T_in=1)

            # 2. Embed Sequence -> full_seq_embeddings (B, T*N, D)
            embedded_tokens_list = []
            for t in range(T):
                # _embed_frame uses parameters (patch_embed, pos_embed, temporal_embed)
                embedded_tokens_list.append(self._embed_frame(target_input_frames[:, t, ...], t))
            full_seq_embeddings = torch.cat(embedded_tokens_list, dim=1) # Concatenation should be fine
            # At this point, full_seq_embeddings *should* require grad due to embedding parameters.

            # 3. Create Causal Mask
            attn_mask = self._generate_square_subsequent_mask(T * N, device) # Does not require grad

            # 4. Pass through Transformer Block
            transformer_out = self.transformer_block(
                src=full_seq_embeddings,
                mask=attn_mask
            ) # Uses transformer parameters. Output *should* require grad.

            # 5. Normalize
            transformer_out_norm = self.norm(transformer_out) # Uses norm parameters. Output *should* require grad.

            # 6. Extract Output Embeddings for Prediction
            # T should be T_out here
            output_embeddings = transformer_out_norm.view(B, T, N, D)[:, :, -1, :] # Slicing/view *should* preserve grad. Output *should* require grad.

            # 7. Generate Predictions
            # _predict_frame_from_embedding uses head parameters.
            # Input `output_embeddings.reshape(-1, D)` *should* require grad.
            preds = self._predict_frame_from_embedding(output_embeddings.reshape(-1, D))
            preds = preds.view(B, T_out, C, H, W) # Reshape back. `preds` *should* require grad.

            return preds, None

        else:
            # --- Autoregressive Generation during Inference ---
            generated_frames = []
            # Embed the initial input frame (step 0)
            current_seq_embeddings = self._embed_frame(x[:, 0, ...], 0) # (B, 1*N, D)

            # Ensure no gradients are computed during generation loop
            with torch.no_grad():
                for t in range(T_out):
                    # Current sequence length feeding into transformer
                    current_seq_len_tokens = current_seq_embeddings.shape[1] # (T_in + t) * N

                    # Create causal mask for the current sequence length
                    attn_mask = self._generate_square_subsequent_mask(current_seq_len_tokens, device)

                    # Pass current sequence through Transformer
                    transformer_out = self.transformer_block(src=current_seq_embeddings, mask=attn_mask) # (B, current_seq_len_tokens, D)
                    transformer_out_norm = self.norm(transformer_out) # Normalize

                    # Get embedding corresponding to the *last* input token
                    last_embedding = transformer_out_norm[:, -1, :] # (B, D)

                    # Predict the next frame (frame t+1 in 0-based indexing, or frame t in sequence gen)
                    pred_frame_t = self._predict_frame_from_embedding(last_embedding) # (B, C, H, W)
                    generated_frames.append(pred_frame_t)

                    # If not the last frame to predict, embed it and append
                    if t < T_out - 1:
                        # Embed the predicted frame for the *next* time step index (t+1)
                        # The temporal embedding index should be t+1 (assuming T_in=1)
                        next_token_embeddings = self._embed_frame(pred_frame_t, t + T_in) # (B, 1*N, D)

                        # Append to the sequence for the next iteration
                        current_seq_embeddings = torch.cat([current_seq_embeddings, next_token_embeddings], dim=1)

            # Stack generated frames
            preds = torch.stack(generated_frames, dim=1) # (B, T_out, C, H, W)
            return preds, None
    
    def objective(self, preds, labels):
        """
        objective loss calculation for the transformer. Differs from other implementations in the
        inclusion of an optional difference loss term for temporal dynamics.
        """
        use_diff_loss = self.arch_conf.use_diff_loss
        lambda_diff = self.arch_conf.lambda_diff

        # --- 1. Calculate the base loss ---
        base_loss = self.compute_loss(preds, labels, choice=self.loss_func)
        total_loss = base_loss

        # --- 2. Calculate the difference loss (if enabled) ---
        if use_diff_loss and preds.shape[1] > 1: # sequence length needs to be greater than 1
            print("[DEBUG] Using diff loss")
            diff_loss = torch.tensor(0.0, device=preds.device, requires_grad=True) # Initialize as zero tensor on correct device
            # Calculate differences between consecutive time steps
            # preds shape: (B, T, C, H, W)
            pred_diff = preds[:, 1:, ...] - preds[:, :-1, ...]  # Shape: (B, T-1, C, H, W)
            label_diff = labels[:, 1:, ...] - labels[:, :-1, ...] # Shape: (B, T-1, C, H, W)

            # Calculate MSE loss on the differences.
            diff_loss_fn = nn.MSELoss()
            diff_loss = diff_loss_fn(pred_diff, label_diff)

            # Combine the base loss and the weighted difference loss
            total_loss = base_loss + lambda_diff * diff_loss

        # --- 3. Logging ---
        log_prefix = 'train' if self.training else 'val'

        self.log(f'{log_prefix}/base_loss', base_loss,
                 on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)

        if use_diff_loss and preds.shape[1] > 1:
            self.log(f'{log_prefix}/diff_loss', diff_loss,
                     on_step=False, on_epoch=True, sync_dist=True, prog_bar=False)
            # Log the lambda value used, helpful for tracking experiments
            self.log('hyperparameters/lambda_diff', lambda_diff,
                     on_step=False, on_epoch=True, sync_dist=True)
 
        return {"loss": total_loss}

    def shared_step(self, batch, batch_idx):
        """
        Shared logic for training/validation steps using the Transformer.
        """
        samples, labels = batch # samples shape (B, T_in, C, H, W), labels shape (B, T_out, C, H, W)
        preds, _ = self.forward(samples, labels) # preds shape (B, T_out, C, H, W)
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
        
        # Append predictions
        self.test_results[mode]['nf_pred'].append(preds_np)
        self.test_results[mode]['nf_truth'].append(labels_np)