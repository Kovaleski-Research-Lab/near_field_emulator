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
        
        super().__init__(model_config, fold_idx)
        
    def create_architecture(self):
        """
        Transformer-specific layers and stuff
        """
        C = 2 # Input channels (real, imag)
        H, W = self.input_height, self.input_width
        self.padded_H = H + get_padding_2d((H, W), self.patch_size)[2] + get_padding_2d((H, W), self.patch_size)[3]
        self.padded_W = W + get_padding_2d((H, W), self.patch_size)[0] + get_padding_2d((H, W), self.patch_size)[1]
        self.grid_size = (self.padded_H // self.patch_size, self.padded_W // self.patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # --- Layers ---
        # 1. Patch Embedding
        self.patch_embed = nn.Conv2d(C, self.embed_dim,
                                     kernel_size=self.patch_size,
                                     stride=self.patch_size)

        # 2. Positional Embeddings
        # Spatial: Learnable embedding for each patch position
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        # Temporal: Learnable embedding for each time step (use max sequence length)
        self.temporal_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, self.embed_dim)) # +1 just in case

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=int(self.embed_dim * self.mlp_ratio),
            dropout=self.dropout,
            activation=F.gelu, # GELU is common in Transformers
            batch_first=True # Expect (batch, seq, feature)
        )
        
        # Split depth between encoder and decoder
        num_encoder_layers = self.depth // 2
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
             d_model=self.embed_dim,
             nhead=self.num_heads,
             dim_feedforward=int(self.embed_dim * self.mlp_ratio),
             dropout=self.dropout,
             activation=F.gelu,
             batch_first=True # Expect (batch, seq, feature)
        )
        num_decoder_layers = self.depth - num_encoder_layers
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 5. Decoder Query Embeddings (learnable tokens to prompt the decoder)
        # One query token per output time step
        self.decoder_query_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        self.decoder_temporal_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))

        # 6. Prediction Head
        # Use an MLP to project frame embedding (D) to all patch embeddings (N*D)
        # then another linear layer to project each patch embedding (D) to patch pixels (P*P*C)
        self.prediction_head_mlp = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.num_patches * self.embed_dim),
            nn.GELU() # nonlinearity
        )
        self.patch_projection = nn.Linear(self.embed_dim, self.patch_size * self.patch_size * C)

        # 7. Unpatching Layer (Optional but good for reversing Conv2d patch embed)
        # This is a simplified inverse; a more complex MLP could also work
        # self.unpatch_layer = nn.ConvTranspose2d(self.embed_dim, C,
        #                                         kernel_size=self.patch_size,
        #                                         stride=self.patch_size)

        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.temporal_embed, std=.02)
        nn.init.trunc_normal_(self.decoder_query_embed, std=.02)
        nn.init.trunc_normal_(self.decoder_temporal_embed, std=.02)
        self.apply(self._init_weights) # Initialize other weights
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x, meta=None):
        """
        Forward pass through the Transformer.
        x shape: (batch, input_seq_len, channels=2, height, width)
        """
        B, T_in, C, H, W = x.shape
        
        # --- Input Processing ---
        # 1. Pad input to be divisible by patch size
        padding = get_padding_2d((H, W), self.patch_size)
        x_padded = F.pad(x.view(B * T_in, C, H, W), padding, mode='constant', value=0)
        # x_padded shape: (B * T_in, C, padded_H, padded_W)

        # 2. Patch Embedding
        # Input: (B * T_in, C, padded_H, padded_W)
        # Output: (B * T_in, embed_dim, grid_H, grid_W)
        x_patch = self.patch_embed(x_padded)
        _, D, grid_H, grid_W = x_patch.shape
        N = grid_H * grid_W # Number of patches

        # 3. Flatten patches and add positional embeddings
        # Output: (B * T_in, embed_dim, N) -> (B * T_in, N, embed_dim)
        x_patch_flat = x_patch.flatten(2).permute(0, 2, 1)

        # Add spatial positional embedding (broadcasts)
        # Need shape (1, N, D)
        x_patch_pos = x_patch_flat + self.pos_embed # (B*T_in, N, D)

        # Reshape to include time dimension for temporal embedding
        # Output: (B, T_in, N, D)
        x_time = x_patch_pos.view(B, T_in, N, D)

        # Add temporal positional embedding (broadcasts)
        # Need shape (1, T_in, 1, D) -> use slicing
        x_time_pos = x_time + self.temporal_embed[:, :T_in, :].unsqueeze(2)

        # Flatten time and patch dimensions for Transformer input
        # Output: (B, T_in * N, D)
        transformer_input = x_time_pos.view(B, T_in * N, D)

        # --- Transformer Encoder ---
        # Input shape needs to be (batch, seq, feature) for batch_first=True
        # Output: (B, T_in * N, D)
        memory = self.transformer_encoder(transformer_input)

        # --- Transformer Decoder ---
        # Prepare decoder query input
        # Input: (1, T_out, D) -> repeat for batch -> (B, T_out, D)
        # T_out is self.seq_len (the desired output sequence length)
        decoder_input = self.decoder_query_embed.repeat(B, 1, 1)
        decoder_input = decoder_input + self.decoder_temporal_embed

        # Decoder expects target, memory. Both (batch, seq, feature)
        # Output: (B, T_out, D)
        decoder_output = self.transformer_decoder(tgt=decoder_input, memory=memory)

        B, T_out = decoder_output.shape[0], decoder_output.shape[1]
        D = self.embed_dim
        C = 2 # Assuming C=2 from context
        N = self.num_patches # N = grid_H * grid_W = 121

        # --- Prediction Head (Revised) ---
        # 1. Project frame embeddings (D) to all patch embeddings (N*D)
        # Input: (B, T_out, D)
        projected_to_all_patches = self.prediction_head_mlp(decoder_output) # Output: (B, T_out, N * D)

        # 2. Reshape to separate the patch dimension
        # Output: (B, T_out, N, D)
        patch_embeddings = projected_to_all_patches.view(B, T_out, N, D)

        # 3. Project each patch's embedding (D) to its flattened pixels (P*P*C)
        # Apply linear layer to the last dimension (D)
        # Input: (B, T_out, N, D)
        # To apply linear efficiently, reshape: (B * T_out * N, D)
        pred_patches_flat = self.patch_projection(patch_embeddings.view(-1, D)) # Output: (B * T_out * N, P*P*C)

        # Reshape back to include B, T_out, N dimensions
        # Output: (B, T_out, N, P*P*C) - THIS is the tensor needed before the problematic view
        pred_patches_flat = pred_patches_flat.view(B, T_out, N, self.patch_size * self.patch_size * C)


        # --- Unpatching ---
        # Now pred_patches_flat has shape (B, T_out, N, P*P*C)
        # Total elements: B * T_out * N * P * P * C = 240 * 121 * 512 = 14,876,160. This matches the target.

        # Reshape for the 6D view operation: (B * T_out, N, P*P*C)
        pred_patches_flat_reshaped = pred_patches_flat.view(B * T_out, N, -1)

        # Reshape into grid structure: (B * T_out, grid_H, grid_W, P*P*C)
        # Note: N must be grid_H * grid_W
        pred_patches_grid = pred_patches_flat_reshaped.view(B * T_out, self.grid_size[0], self.grid_size[1], self.patch_size * self.patch_size * C)

        # Reshape into the 6D structure needed for permuting patches
        # Input: (B*T_out, grid_H, grid_W, P*P*C)
        # Output: (B*T_out, grid_H, grid_W, P, P, C)
        # This reshape requires P*P*C to be ordered correctly (e.g., C-last)
        pred_patches = pred_patches_grid.view(B * T_out, self.grid_size[0], self.grid_size[1], self.patch_size, self.patch_size, C) # This is the problematic line (201 approx) - should work now

        # Permute to put C channel first, then merge patch dimensions
        # Output: (B * T_out, C, grid_H, P, grid_W, P)
        pred_patches = pred_patches.permute(0, 5, 1, 3, 2, 4)

        # Reshape to final padded image size by merging grid and patch dims
        # Output: (B * T_out, C, H_pad, W_pad)
        preds_padded = pred_patches.reshape(B * T_out, C, self.padded_H, self.padded_W)

        # --- Final Output ---
        # Crop padding to original size
        padding = get_padding_2d((self.input_height, self.input_width), self.patch_size)
        crop_h_start = padding[2]
        crop_h_end = self.padded_H - padding[3]
        crop_w_start = padding[0]
        crop_w_end = self.padded_W - padding[1]

        preds = preds_padded[:, :, crop_h_start:crop_h_end, crop_w_start:crop_w_end]
        # preds shape: (B * T_out, C, H, W)

        # Reshape to final format: (B, T_out, C, H, W)
        preds_final = preds.view(B, T_out, C, self.input_height, self.input_width)

        return preds_final, None
    
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
        diff_loss = torch.tensor(0.0, device=preds.device) # Initialize as zero tensor on correct device
        if use_diff_loss and preds.shape[1] > 1: # sequence length needs to be greater than 1
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
        # T_out should == self.seq_len

        # Forward pass
        preds, _ = self.forward(samples) # preds shape (B, T_out, C, H, W)

        # Compute loss
        # Ensure labels have the expected shape if T_in != T_out
        # Loss function expects (B, T_out, ...) or flattened equivalent
        # Our base class objective likely handles this if preds/labels match dims
        loss_dict = self.objective(preds, labels)
        loss = loss_dict['loss']

        # Return loss and predictions (already in correct B, T, C, H, W format)
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