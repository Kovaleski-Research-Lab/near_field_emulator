import torch
from .WavePropModel import WavePropModel

class WaveLSTM(WavePropModel):
    """Near Field Response Time Series Prediction Model  
    Architecture: LSTM"""
    def __init__(self, model_config, fold_idx=None):
        super().__init__(model_config, fold_idx)

    def create_architecture(self):
        self.arch_conf = self.conf.lstm
        i_dims = self.arch_conf.i_dims
            
        self.arch = torch.nn.LSTM(input_size=i_dims,
                                    hidden_size=self.arch_conf.h_dims,
                                    num_layers=self.arch_conf.num_layers,
                                    batch_first=True)
            
        self.linear = torch.nn.Sequential(
                        torch.nn.Linear(self.arch_conf.h_dims, i_dims),
                        torch.nn.Tanh())
        
    def forward(self, x, meta=None):
        batch, seq_len, input = x.size()
        if meta is None: # Init hidden and cell states if not provided
            h, c = self.init_hidden(batch)
            meta = (h, c)          
            
        # flatten spatial and r/i dims - (batch_size, seq_len, input_size)
        x = x.view(batch, seq_len, -1)
        
        predictions = [] # to store each successive pred as we pass through
        
        if self.io_mode == 'one_to_many':
            # first timestep: use our single slice input
            lstm_out, meta = self.arch(x, meta)
            pred = self.linear(lstm_out)
            predictions.append(pred) # t+1 (or + delta split)
            
            # remaining t: no input, we pass 0's as dummy vals
            for t in range(1, self.seq_len):
                dummy_input = torch.zeros_like(x)
                lstm_out, meta = self.arch(dummy_input, meta)
                pred = self.linear(lstm_out)
                predictions.append(pred)
            
        elif self.io_mode == 'many_to_many':
            if self.autoreg:
                # Use first timestep
                current_input = x[:, 0]  # Keep seq_len dim with size 1
                current_input = current_input.unsqueeze(1)
                lstm_out, meta = self.arch(current_input, meta)
                pred = self.linear(lstm_out)
                predictions.append(pred)
                
                # Generate remaining predictions using previous outputs
                for t in range(1, self.seq_len):
                    # Use previous prediction as input
                    lstm_out, meta = self.arch(pred, meta)
                    pred = self.linear(lstm_out)
                    predictions.append(pred)
            else: # teacher forcing
                for t in range(self.seq_len):
                    current_input = x[:, t] # ground truth at t
                    lstm_out, meta = self.arch(current_input, meta)
                    pred = self.linear(lstm_out)
                    predictions.append(pred)
                
        else:
            # other io modes not currently implemented
            return NotImplementedError(f'Recurrent input-output mode "{self.io_mode}" is not implemented.')
        
        # Stack predictions along sequence dimension
        predictions = torch.cat(predictions, dim=1)
        
        return predictions, meta
        
    def shared_step(self, batch, batch_idx):
        samples, labels = batch
        
        # extract sizes - input sequence could be len=1 if 12M
        batch_size, input_seq_len, r_i, xdim, ydim = samples.size()
        

        # flatten spatial and r/i dims - (batch_size, seq_len, input_size)
        samples = samples.view(batch_size, input_seq_len, -1)
        labels = labels.view(batch_size, self.seq_len, -1)
        
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
    
    def init_hidden(self, batch_size):
        h = torch.zeros(self.arch_conf.num_layers, 
                        batch_size, self.arch_conf.h_dims).to(self._device)
        c = torch.zeros(self.arch_conf.num_layers, 
                        batch_size, self.arch_conf.h_dims).to(self._device)
        return (h, c)
        
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