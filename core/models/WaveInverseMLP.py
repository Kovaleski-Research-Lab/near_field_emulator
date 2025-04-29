#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import sys
import torch
import numpy as np

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
from .WaveResponseModel import WaveResponseModel
from .WaveForwardMLP import WaveForwardMLP

sys.path.append("../")

class WaveInverseMLP(WaveResponseModel):
    """Near Field Response Prediction Model - Inverse Design"""
    def __init__(self, model_config, fold_idx=None):
        self.frozen_forward_model = None
        # tandem init - special
        if model_config.inverse_strategy == 1: # tandem
            forward_model_ckpt_path = model_config.forward_ckpt
            
            try:
                self.forzen_forward_model = WaveForwardMLP.load_from_checkpoint(
                    forward_model_ckpt_path,
                    map_location='cpu'
                )
            except Exception as e:
                
                print(f"Error loading forward model checkpoint: {e}")
                print("Attempting to load hparams separately...")
                try:
                    # Fallback: Load hparams and instantiate manually
                    ckpt = torch.load(forward_model_ckpt_path, map_location='cpu')
                    forward_hparams = ckpt.get('hyper_parameters', {})
                    self.frozen_forward_model = WaveForwardMLP(model_config=forward_hparams.get('conf', forward_hparams))
                    self.frozen_forward_model.load_state_dict(ckpt['state_dict'])
                except Exception as e2:
                    raise RuntimeError("Failed to load forward model checkpoint using both methods")
                
            self.frozen_forward_model.eval()
            for param in self.frozen_forward_model.parameters():
                param.requires_grad = False
        
        super().__init__(model_config, fold_idx)
        self.output_size = self.num_design_conf
        
    def create_architecture(self):
        # specific strategies for this model
        self.strat = None
        if self.conf.inverse_strategy == 0:
            self.strat = 'naive'
        elif self.conf.inverse_strategy == 1:
            self.strat = 'tandem'
            input_dim = self.near_field_dim**2
            if self.frozen_forward_model is not None:
                self.frozen_forward_model = self.frozen_forward_model.to(self.device)
            else:
                raise RuntimeError("Tandem strategy selected but frozen_forward_model is not loaded.")
        else:
            raise ValueError(f"Inverse strategy {self.conf.inverse_strategy} not recongized.")
        
        # initialize the architecture for inverse MLP
        self.cvnn = self.build_mlp(self.near_field_dim**2, self.conf.cvnn)
        
    def forward(self, designs, near_fields):
        # going from fields to design
        batch_size = near_fields.size(0)
        nf_complex = torch.complex(near_fields[:, 0, :, :], near_fields[:, 1, :, :])
        nf_complex = nf_complex.view(batch_size, -1)
        predicted_designs = self.cvnn(nf_complex)
        return predicted_designs.real # real component of the complex linear output
  
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # setup specified scheduler
        if self.lr_scheduler == 'ReduceLROnPlateau':
            choice = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                                    factor=0.5, patience=5, 
                                                                    min_lr=1e-6, threshold=0.001, 
                                                                    cooldown=2)
        elif self.lr_scheduler == 'CosineAnnealingLR':
            choice = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                T_max=100,
                                                                eta_min=1e-6)
        elif self.lr_scheduler == 'None':
            return optimizer
        else:
            raise ValueError(f"Unsupported LR scheduler: {self.lr_scheduler}")
        
        scheduler = {
            'scheduler': choice,
            'interval': 'epoch',
            'monitor': 'val_loss',
            'frequency': 1
        }

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def objective(self, batch, predictions):
        target_fields, target_designs = batch
        predicted_designs = predictions
        
        loss_dict = {}
        
        if self.strat == 'naive':
            design_loss = self.compute_loss(predicted_designs, target_designs, choice=self.loss_func)
            loss_dict['loss'] = design_loss
            
            # Calculate other metrics on designs
            other_metrics = ['mse'] # Add others
            for key in other_metrics:
                if key != self.loss_func:
                    loss = self.compute_loss(predicted_designs, target_designs, choice=key)
                    loss_dict[key] = loss
                    
        elif self.strat == 'tandem':
            # 1. pass predicted designs through frozen forward model
            with torch.no_grad():
                reconstructed_output = self.frozen_forward_model(predicted_designs, None)
            # output is [B, 2, H, W]
            recon_fields_real = reconstructed_output[:, 0, :, :]
            recon_fields_imag = reconstructed_output[:, 1, :, :]
            
            # 2. Compare reconstructed fields to target ground truth fields
            target_fields_real = target_fields[:, 0, :, :]
            target_fields_imag = target_fields[:, 1, :, :]
            
            # calculate the loss on fields
            loss_real = self.compute_loss(recon_fields_real, target_fields_real, choice=self.loss_func)
            loss_imag = self.compute_loss(recon_fields_imag, target_fields_imag, choice=self.loss_func)
            field_loss = loss_real + loss_imag
            loss_dict['loss'] = field_loss
            
            # 3. Calculate other metrics on fields
            other_metrics = ['mse', 'ssim', 'psnr']
            for key in other_metrics:
                if key != self.loss_func:
                     loss_r = self.compute_loss(recon_fields_real, target_fields_real, choice=key)
                     loss_i = self.compute_loss(recon_fields_imag, target_fields_imag, choice=key)
                     loss_dict[key] = loss_r + loss_i # Store combined metric
                     
        else:
            raise ValueError(f"Unknown strategy '{self.strat}' in objective.")
        
        final_loss_dict = {"loss": loss_dict.get('loss', torch.tensor(float('nan'), device=self.device))}
        for key, value in loss_dict.items():
            if key != 'loss':
                final_loss_dict[key] = value if value is not None else torch.tensor(float('nan'), device=self.device)

        return final_loss_dict

    def shared_step(self, batch, batch_idx):
        near_fields, designs = batch
        preds = self.forward(designs, near_fields)
        return preds
        
    def organize_testing(self, predictions, batch, batch_idx, dataloader_idx):
        target_fields, target_designs = batch
        predicted_designs = predictions
        
        # Store predicted designs and ground truth designs
        preds_to_store = predicted_designs.detach().cpu().numpy()
        truths_to_store = target_designs.detach().cpu().numpy()
        
        mode = 'valid' if dataloader_idx == 0 else 'train'
        # store design parameters
        design_pred_key = 'design_pred'
        design_truth_key = 'design_truth'
        
        if mode not in self.test_results: self.test_results[mode] = {}
        if design_pred_key not in self.test_results[mode]: self.test_results[mode][design_pred_key] = []
        if design_truth_key not in self.test_results[mode]: self.test_results[mode][design_truth_key] = []

        self.test_results[mode][design_pred_key].append(preds_to_store)
        self.test_results[mode][design_truth_key].append(truths_to_store)
        
        # store fields if we're doing tandem model
        if self.strat == 'tandem':
            field_pred_key = 'nf_pred'
            field_truth_key = 'nf_truth'
            if field_pred_key not in self.test_results[mode]: self.test_results[mode][field_pred_key] = []
            if field_truth_key not in self.test_results[mode]: self.test_results[mode][field_truth_key] = []

            with torch.no_grad():
                # handle forward model output to get [B, 2, H, W]
                reconstructed_output = self.frozen_forward_model(predicted_designs.float().to(self.device), None)
                recon_fields_np = torch.stack([reconstructed_output.real, reconstructed_output.imag], dim=1).detach().cpu().numpy()
                
            target_fields_np = target_fields.detach().cpu().numpy()
            self.test_results[mode][field_pred_key].append(recon_fields_np)
            self.test_results[mode][field_truth_key].append(target_fields_np)

    def on_test_end(self):
        """Concatenate results from all batches for all stored keys."""
        for mode in ['train', 'valid']:
            if mode in self.test_results:
                 for key in list(self.test_results[mode].keys()): # Iterate over keys found
                     if self.test_results[mode][key]: # Check if list is not empty
                         try:
                             # Ensure all elements are numpy arrays before concatenating
                             if all(isinstance(x, np.ndarray) for x in self.test_results[mode][key]):
                                 self.test_results[mode][key] = np.concatenate(self.test_results[mode][key], axis=0)
                             else:
                                 print(f"Warning: test_results['{mode}']['{key}'] contains non-ndarray elements.")
                                 self.test_results[mode][key] = np.array([]) # Clear if mixed types
                         except ValueError as e:
                              print(f"Error concatenating test results for {mode}/{key}: {e}")
                              # This can happen if arrays have inconsistent shapes beyond axis 0
                              self.test_results[mode][key] = self.test_results[mode][key] # Keep as list of arrays
                     else:
                         print(f"No test results found or list empty for mode '{mode}', key '{key}'.")
                         self.test_results[mode][key] = np.array([]) # Set to empty array