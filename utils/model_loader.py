#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import yaml
import torch
import logging

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------

#from utils.mapping import get_model_type
from core.models import *

def select_model(model_config, fold_idx=None):
    logging.debug("select_model.py - Selecting model") 
    #model_type = get_model_type(pm.arch)
    model_type = model_config.arch
    if model_type == 'autoencoder': # autoencoder pretraining
        network = Autoencoder(model_config, fold_idx)
    elif model_type == 'mlp' or model_type == 'cvnn':
        network = WaveForwardMLP(model_config, fold_idx)
    elif model_type == "inverse":
        network = WaveInverseMLP(model_config, fold_idx)
    # mode lstm is just the lstm but on pre-encoded data
    elif model_type == 'lstm':
        network = WaveLSTM(model_config, fold_idx)
    elif model_type == 'modelstm':
        network = WaveModeLSTM(model_config, fold_idx)
    elif model_type == 'convlstm':
        network = WaveConvLSTM(model_config, fold_idx)
    elif model_type == 'ae-lstm':
        network = WaveAELSTM(model_config, fold_idx)
    elif model_type == 'ae-convlstm':
        network = WaveAEConvLSTM(model_config, fold_idx)
    elif model_type == 'diffusion':
        network = WaveDiffusion(model_config, fold_idx)
    elif model_type == 'transformer':
        network = WaveTransformer(model_config, fold_idx)
    else:
        raise NotImplementedError("Model type not recognized.")

    assert network is not None

    return network
