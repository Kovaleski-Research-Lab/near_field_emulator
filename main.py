import os
import shutil
import argparse
import yaml
import torch

from core import train, preprocess_data, compile_data, modes
from conf.schema import load_config
from evaluation import eval_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "config.yaml specifiying experiment parameters")
    args = parser.parse_args()
    
    # Load parameters from the specified config YAML
    #params = yaml.load(open(args.config), Loader = yaml.FullLoader).copy()
    #directive = params['directive'] # WALL-E reference?
    
    # Load the MainConfig object
    config = load_config(args.config)
    directive = config.directive
    
    # set precision
    torch.set_float32_matmul_precision(config.trainer.matmul_precision)
    
    if directive == 0:
        print("Training model...\n")
        _, _ = train.run(config)
        
    elif directive == 1:
        print('Evaluating model...\n')
        eval_model.run(config)
        
    elif directive == 2:
        print('Training model...\n')
        _, data_module =train.run(config)
        print('\nTraining complete. Evaluating model...\n')
        eval_model.run(config, data_module)
        
    elif directive == 3:
        if config.pipeline is not None:
            print("Running multi-phase pipeline...\n")
            # Pipeline is already constructed and validated by schema
            trained_models, data_module = train.run(config)
            print("\nPipeline training complete. Evaluating final model...\n")
            eval_model.run(config, data_module, trained_models)
        else:
            raise ValueError("Directive 3 requires a 'pipeline' section in the configuration.")
    
    elif directive == 4:
        raise NotImplementedError('Loading results not fully implemented yet.')
    
    elif directive == 5:
        raise NotImplementedError('MEEP simulation process not fully implemented yet.')
    
    elif directive == 6:
        print('Preprocessing DFT volumes...')
        preprocess_data.run(config)
        print('Compiling data into .pt file...')
        compile_data.run(config)
        
    elif directive == 7:
        print(f"Encoding {config.model.modelstm.method} modes...")
        modes.run(config)
        
    elif directive == 8:
        print(f'Reconstructing {config.model.modelstm.method} modes...')
        modes.run(config)
        
    else:
        raise NotImplementedError(f'config.yaml: directive: {directive} is not a valid directive.')
