import torch
import torch.nn as nn
from .WaveModel import WaveModel
from .ConvLSTM import ConvLSTM

class WaveConvLSTM(WaveModel):
    """Near Field Response Time Series Prediction Model  
    Architecture: ConvLSTM"""
    def __init__(self, model_config, fold_idx=None):
        super().__init__(model_config, fold_idx)

    def create_architecture(self):
        self.arch_conf = self.conf.convlstm
        
        self.seq_len = self.conf.seq_len
        self.kernel_size = self.arch_conf.kernel_size
        self.padding = self.arch_conf.padding
        self.out_channels = self.arch_conf.out_channels
        self.num_layers = self.arch_conf.num_layers
        self.in_channels = self.arch_conf.in_channels
        self.spatial = self.arch_conf.spatial
        
        # Create single ConvLSTM layer
        self.arch = ConvLSTM(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            seq_len=self.seq_len,
            kernel_size=self.kernel_size,
            padding=self.padding,
            frame_size=(self.spatial, self.spatial)
        )
        
        # conv reduction + activation to arrive back at real/imag
        self.linear = nn.Sequential(
            nn.Conv2d(self.out_channels, 2, kernel_size=1),
            nn.Tanh(),
        )
        
    def forward(self, x, meta=None):
        batch, seq_len, r_i, xdim, ydim = x.size()
        x = x.view(batch, seq_len, self.in_channels, 
                    self.spatial, self.spatial)
        
        # invoke for specified mode (i.e. many_to_many)
        preds, meta = self.arch(x, meta, mode=self.io_mode, 
                                    autoregressive=self.autoreg)
        
        # reshape for conv
        #b, s, ch, he, w = lstm_out.size()
        #lstm_out = lstm_out.view(b * s, ch, he, w)
        # apply conv + tanh
        #preds = self.linear(lstm_out)
        # reshape back
        #preds = preds.view(b, s, 2, he, w)

        return preds, meta
        
    def shared_step(self, batch, batch_idx):
        samples, labels = batch
        
        # extract sizes - input sequence could be len=1 if 12M
        batch_size, input_seq_len, r_i, xdim, ydim = samples.size()
        
        # Forward pass
        preds, _ = self.forward(samples)
        
        # Compute loss
        loss_dict = self.objective(preds, labels)
        loss = loss_dict['loss']
        
        # reshape preds for metrics
        if self.io_mode == "one_to_many":
            preds = preds.view(batch_size, self.seq_len, r_i, xdim, ydim)
        elif self.io_mode == "many_to_many":
            preds = preds.view(batch_size, self.seq_len, r_i, xdim, ydim)
        else:
            # other modes not implemented
            raise NotImplementedError

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