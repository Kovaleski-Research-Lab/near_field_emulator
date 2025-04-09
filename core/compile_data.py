#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import logging
import yaml
import os
import sys
from pytorch_lightning import seed_everything

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
sys.path.append('../')
from core import datamodule as dm

def run(conf):
    """This function finishes the preprocessing pipeline for the data,  
       fully loading it into the PyTorch format expected by the dataloader.  
       It operates on the preprocessed pickle files separated by train/valid.  
       in the preprocessed_data directory.
    """
    logging.basicConfig(level=logging.DEBUG)
    seed_everything(1337)

    # for accessing the preprocessed data
    if conf.deployment == 0:  # we are using local compute
        path_output = conf.paths.data
    elif conf.deployment == 1: # we are launching kubernetes jobs
        path_output = conf.kube.data_job['paths']['data']['preprocessed_data']
    
    model_type = conf.model.arch
    parameter = conf.physics.material_parameter
    
    train_path = os.path.join(path_output, 'train')
    valid_path = os.path.join(path_output, 'valid')
    
    # Load data
    if model_type == 'mlp' or model_type == 'cvnn':
        save_path = os.path.join(path_output, 'dataset_nobuffer.pt')
        if os.path.exists(save_path):
            raise FileExistsError(f"Output file {save_path} already exists!")
        dm.load_pickle_data(train_path, valid_path, save_path, parameter=parameter, arch='mlp')
    else: # LSTM
        # convert conf.data.wv_dict[conf.data.wv_preprocess] from float to string and strip the decimal
        wavelength = str(conf.data.wv_dict[conf.data.wv_preprocess]).replace('.', '')
        save_path = os.path.join(path_output, f'dataset_{wavelength}.pt')
        logging.debug(f"Save path: {save_path}")
        logging.debug(f"Save directory exists: {os.path.exists(os.path.dirname(save_path))}")
        logging.debug(f"Save directory writable: {os.access(os.path.dirname(save_path), os.W_OK)}")
        #if os.path.exists(save_path):
        #    raise FileExistsError(f"Output file {save_path} already exists!")
        dm.load_pickle_data(train_path, valid_path, save_path, parameter=parameter, arch='lstm')