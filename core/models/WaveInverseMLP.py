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

sys.path.append("../")

class WaveInverseMLP(WaveResponseModel):
    """Near Field Response Prediction Model - Inverse Design"""
    def __init__(self, model_config, fold_idx=None):
        super().__init__(model_config, fold_idx)
        
    def create_architecture(self):
        # specific strategies for this model
        self.strat = None
        if self.conf.inverse_strategy == 0:
            self.strat = 'naive'
        elif self.conf.inverse_strategy == 1:
            self.strat = 'tandem'
        
        self.output_size = 1
        self.cvnn = self.build_mlp(self.near_field_dim**2, self.conf.cvnn)
        
    def forward(self, designs, near_fields):
        if self.name == 'cvnn':
            if self.strat == 'naive':
                # going from fields to design
                batch_size = near_fields.size(0)
                nf_complex = torch.complex(near_fields[:, 0, :, :], near_fields[:, 1, :, :])
                nf_complex = nf_complex.view(batch_size, -1)
                output = self.cvnn(nf_complex)
                return output
            elif self.strat == 'tandem':
                raise NotImplementedError("Not quite there yet - Tandem model")
        
        else:
            # safeguard
            raise NotImplementedError("Dual MLPs or methods other than CVNN for inverse strategy not yet implemented.")
  
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
        near_fields, radii = batch
        labels = radii  
        preds = predictions.real
        radii_loss = self.compute_loss(near_fields, preds, labels, choice=self.loss_func)
        
        choices = {
            'mse': None,
            'resim': None
        }
        for key in choices:
            if key != self.loss_func:
                loss = self.compute_loss(near_fields, preds, labels, choice=key)
                choices[key] = loss
                
        return {"loss": radii_loss, **choices}


    def shared_step(self, batch, batch_idx):
        near_fields, designs = batch
        preds = self.forward(designs, near_fields)
        return preds
        
    def organize_testing(self, predictions, batch, batch_idx, dataloader_idx):
        near_fields, designs = batch
        
        preds = predictions
        truths = designs
        
        # Store predictions and ground truths for analysis after testing
        if dataloader_idx == 0:  # val dataloader
            self.test_results['valid']['nf_pred'].append(preds)
            self.test_results['valid']['nf_truth'].append(truths)
        elif dataloader_idx == 1:  # train dataloader
            self.test_results['train']['nf_pred'].append(preds)
            self.test_results['train']['nf_truth'].append(truths)
        else:
            raise ValueError(f"Invalid dataloader index: {dataloader_idx}")

    def on_test_end(self):
        # Concatenate results from all batches
        for mode in ['train', 'valid']:
            self.test_results[mode]['nf_pred'] = np.concatenate(self.test_results[mode]['nf_pred'], axis=0)
            self.test_results[mode]['nf_truth'] = np.concatenate(self.test_results[mode]['nf_truth'], axis=0)