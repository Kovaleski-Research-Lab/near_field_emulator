#--------------------------------
# Import: Basic Python Libraries
#--------------------------------

import os
import yaml
import torch
import logging
#from pytorch_lightning.strategies import DDPStrategy
#from pytorch_lightning import Trainer, seed_everything
#from pytorch_lightning.callbacks import ModelCheckpoint

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
        network = WaveMLP(model_config, fold_idx)
    # mode lstm is just the lstm but on epre-encoded data
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
    else:
        raise NotImplementedError("Model type not recognized.")

    '''if config.trainer.load_checkpoint:
         
        checkpoint_path = os.path.join(config.paths.root, config.paths.checkpoint)
        checkpoint = os.listdir(checkpoint_path)[0]
        checkpoint = os.path.join(checkpoint_path, checkpoint)
        print(checkpoint)
        
        state_dict = torch.load(checkpoint)['state_dict']
        #network.load_from_checkpoint(pm.path_checkpoint,
        #                                   params = (pm.params_model, pm.params_propagator),
        #                                   strict = True)
        network.load_state_dict(state_dict, strict=True)'''

    assert network is not None

    return network
