"""
Purpose: ConvLSTM Implementation - continuation from meta_atom_rnn/utils/network/convlstm.py
Authors: Andy, Ethan
"""


#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import sys
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

sys.path.append("../")


class ConvLSTMCell(nn.Module):
    """
    ConvLSTMCell class
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, frame_size):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.frame_size = frame_size

        # primary conv layer -> processes concatenated input and prev hidden state
        # output channels = 4 * out_channels (i, f, c, o)
        self.conv = nn.Conv2d(in_channels=self.in_channels + self.out_channels,
                              out_channels=4 * self.out_channels,
                              kernel_size=self.kernel_size, padding=self.padding)

        # learnable weights for input, output, and forget gates
        self.W_ci = nn.Parameter(torch.zeros(self.out_channels, *self.frame_size)) 
        self.W_co = nn.Parameter(torch.zeros(self.out_channels, *self.frame_size))
        self.W_cf = nn.Parameter(torch.zeros(self.out_channels, *self.frame_size))
        
        # init weights
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x, h_prev, c_prev):
        # concat input and prev hidden state
        conv_output = self.conv(torch.cat([x, h_prev], dim=1))

        # split into input, forget, cell, and output gates
        i_conv, f_conv, c_conv, o_conv = torch.chunk(conv_output,
                                                     chunks=4, dim=1)

        # apply sigmoid to input, forget, and cell gates
        input_gate = torch.sigmoid(i_conv + self.W_ci * c_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * c_prev)

        # apply tanh to update the cell state
        c = forget_gate * c_prev + input_gate * torch.tanh(c_conv)

        # apply sigmoid to output gate and calculate output
        output_gate = torch.sigmoid(o_conv + self.W_co * c)
        h = output_gate * torch.tanh(c)

        return h, c


class ConvLSTM(nn.Module):
    """
    ConvLSTM class
    """
    def __init__(self, in_channels, out_channels, seq_len,
                 kernel_size, padding, frame_size):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.target_len = seq_len

        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels,
                                         kernel_size, padding, frame_size)
        
        # for converting hidden states to next step inputs
        '''self.output_proj = nn.Conv2d(
            in_channels=out_channels,  # 64
            out_channels=in_channels,  # 2
            kernel_size=1  # 1x1 conv
        )'''
        self.output_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels // 2, in_channels, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self.smoothing = nn.Sequential(
            nn.ReplicationPad2d(2),
            nn.AvgPool2d(kernel_size=5, stride=1)
        )

    def forward(self, x, meta=None, mode="one_to_many", autoregressive=False):
        batch_size, seq_len, channel, height, width = x.size()

        if meta is None:
            # init hidden and cell states
            h = torch.zeros(batch_size, self.out_channels,
                            height, width, device=x.device)
            c = torch.zeros(batch_size, self.out_channels,
                            height, width, device=x.device)
            meta = (h, c)
        else:
            h, c = meta

        # prepare output tensor
        output = torch.zeros(batch_size, self.target_len,
                             self.out_channels, height, width, device=x.device)

        if mode == "one_to_many":
            # squeeze x to remove seq dim
            x = x.squeeze(dim=1) # [batch, 2, 166, 166]
            # process the single input time step
            h, c = self.convLSTMcell(x, h, c)
            output[:, 0] = h
            
            # generate remaining t's with zero inputs
            dummy_input = torch.zeros_like(x)
            for t in range(1, self.target_len): # 1 through the end
                h, c = self.convLSTMcell(dummy_input, h, c)
                output[:, t] = h
                
        elif mode == "many_to_many":
            # each step has input and output
            if autoregressive: # input is last one's output
                current_input = x[:, 0]
                for t in range(seq_len):
                    h, c = self.convLSTMcell(current_input, h, c)
                    output[:, t] = h
                    # escape hidden representation
                    current_input = self.smoothing(self.output_proj(h))
            else: # input is current timestep - teacher forcing
                for t in range(seq_len):
                    current_input = x[:, t] # [batch, 2, 166, 166]
                    h, c = self.convLSTMcell(current_input, h, c)
                    output[:, t] = h
                    
        # Apply smoothing to final output if needed
        if mode == "one_to_many" or (mode == "many_to_many" and not autoregressive):
            # Reshape output for smoothing
            B, T, C, H, W = output.size()
            output = output.view(B * T, C, H, W)
            output = self.output_proj(output)
            output = self.smoothing(output)
            # Reshape back
            output = output.view(B, T, 2, H, W)
                   
        return output, (h, c)
