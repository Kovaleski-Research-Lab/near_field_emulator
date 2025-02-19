#--------------------------------
# Import: Basic Python Libraries
#--------------------------------
import itertools
import os
import sys
import torch
import logging
import numpy as np
from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, Subset
import pickle
import torch
from tqdm import tqdm
from typing import Dict

#--------------------------------
# Import: Custom Python Libraries
#--------------------------------
sys.path.append('../')
from core import curvature, modes
from utils import mapping


# debugging
#logging.basicConfig(level=logging.DEBUG)

# --------------------------------
# Raw Data Loading
# --------------------------------

class RawDataLoader:
    """
    Loads and caches the raw data from disk. This class is responsible for reading
    the data file (which contains ALL data) only once.
    """
    def __init__(self, conf):
        self.conf = conf
        self.data_cache: Dict[str, dict] = {} # Cache keyed by stage ("fit, "test")
        #self.train_means = None
        #self.train_stds = None
        
    def load(self, stage: str) -> dict:
        if stage in self.data_cache:
            return self.data_cache[stage]
        
        if stage in ["fit", None]:
            data = self._load_data(self.conf.data.wv_train)
            if self.conf.data.normalize:
                # Calculate and store training statistics
                train_means = data['near_fields'].mean(dim=(1,2,3), keepdim=True)
                train_stds = data['near_fields'].std(dim=(1,2,3), keepdim=True)
                # Standardize using these statistics
                data['near_fields'] = (data['near_fields'] - train_means) / train_stds
                print(f"Standardized near fields for training set")
                # save training statistics
                torch.save({
                    'means': train_means,
                    'stds': train_stds
                }, os.path.join(self.conf.paths.results, 'train_stats.pt'))
        elif stage == "test":
            data = self._load_data(self.conf.data.wv_eval)
            if self.conf.data.normalize: # and self.conf.data.wv_train != self.conf.data.wv_eval:
                #if self.train_means is None or self.train_stds is None:
                #    raise ValueError("Training set statistics not available. Must load training data before evaluation data.")
                # Standardize using training statistics
                train_stats = torch.load(os.path.join(self.conf.paths.results, 'train_stats.pt'))
                train_means = train_stats['means']
                train_stds = train_stats['stds']
                data['near_fields'] = (data['near_fields'] - train_means) / train_stds
                print(f"Standardized near fields for validation set using training statistics")
        else:
            raise ValueError(f"Unsupported stage: {stage}")

        self.data_cache[stage] = data
        return data
    
    def _load_data(self, wv_idx):
        if self.conf.model.arch == 'modelstm':
            datapath = self._get_datapath()
            print(f"Loading data from datapath: {datapath}")
            data = torch.load(datapath, weights_only=True)
            data, P, mean_vec = preprocess_svd_data(data, self.conf)
            self.P = P
            self.mean_vec = mean_vec
            return data
        else:
            # If multiple wavelengths are specified, combine them.
            if isinstance(wv_idx, (list, tuple)) and len(wv_idx) > 1:
                data_combined = {'near_fields': [], 'phases': [], 
                                 'derivatives': [], 'radii': [], 
                                 'tag': [], 'wavelength': []}
                for wv in tqdm(wv_idx, desc="Loading data...", ncols=80, file=sys.stderr, mininterval=1.0, dynamic_ncols=True):
                    datapath = self._get_datapath(wv)
                    print(f"Loading data from datapath: {datapath}")
                    wv_data = torch.load(datapath, weights_only=True)
                    num_samples = wv_data['near_fields'].shape[0]
                    for key in wv_data.keys():
                        data_combined[key].append(wv_data[key])
                    # Record the wavelength for each sample.
                    data_combined['wavelength'].append(torch.full((num_samples,), self.conf.data.wv_dict[wv], dtype=torch.float))
                # Concatenate lists into tensors.
                for key in data_combined:
                    data_combined[key] = torch.cat(data_combined[key], dim=0)
                return data_combined
            else:
                datapath = self._get_datapath(wv_idx)
                wv_data = torch.load(datapath, weights_only=True)
                wv_data['wavelength'] = torch.full((wv_data['near_fields'].shape[0],), self.conf.data.wv_dict[wv_idx], dtype=torch.float)
                return wv_data
            
    def _get_datapath(self, wv_idx=None):
        """Based on params, return the correct dataset we'll be using"""
        if not self.conf.data.buffer:
            return os.path.join(self.conf.paths.data, 'preprocessed_data', 'dataset_nobuffer.pt')
        elif self.conf.model.arch == 'modelstm':
            return os.path.join(self.conf.paths.data, 'preprocessed_data', f"dataset_{self.conf.model.modelstm.method}.pt")
        else:
            if wv_idx is None:
                raise ValueError("Wavelength index is required for dataset retrieval")
            wv = str(self.conf.data.wv_dict[wv_idx]).replace('.', '')
            return os.path.join(self.conf.paths.data, 'preprocessed_data', f'dataset_{wv}.pt')
 
# --------------------------------
# Data Processing
# --------------------------------       

class DataProcessor:
    def process(self, raw_data: dict, conf) -> Dataset:
        raise NotImplementedError("process() must be implemented in subclasses")
    
class AutoencoderProcessor(DataProcessor):
    def process(self, raw_data, conf):
        return format_ae_data(raw_data, conf)

class MLPProcessor(DataProcessor):
    def process(self, raw_data, conf):
        if conf.model.interpolate_fields:
            raw_data = interpolate_fields(raw_data)
        # Get transform from model config if it exists, otherwise None
        transform = getattr(conf.model, 'transform', None)
        return WaveMLP_Dataset(
            raw_data, 
            transform, 
            conf.model.mlp_strategy, 
            conf.model.patch_size, 
            buffer=conf.data.buffer
        )
        
class TemporalProcessor(DataProcessor):
    def process(self, data, conf):
        #print(f"\nINITIATING CALL OF format_temporal_data")
        return format_temporal_data(data, conf)
    
def get_processor(conf) -> DataProcessor:
    mapping_processors = {
        'autoencoder': AutoencoderProcessor(),
        'mlp': MLPProcessor(),
        'cvnn': MLPProcessor()
    }
    # Default to temporal processing if the architecture key is not found.
    return mapping_processors.get(conf.model.arch, TemporalProcessor())

# --------------------------------
# Lightning DataModule
# --------------------------------

class NFDataModule(LightningDataModule):
    """
    Generic DataModule that loads the raw data only once and processes it
    according to the model type. The raw data is shared and then formatted via
    a chosen DataProcessor.
    """
    def __init__(self, conf, transform = None):
        super().__init__() 
        logging.debug("datamodule.py - Initializing NF_DataModule")

        self.conf = conf.copy()
        self.directive = conf.directive
        self.model_type = conf.model.arch
        self.n_cpus = conf.data.n_cpus
        self.seed = conf.seed
        self.n_folds = conf.data.n_folds
        self.path_data = conf.paths.data
        self.normalize = conf.data.normalize
        self.batch_size = conf.trainer.batch_size
        self.wv_dict = conf.data.wv_dict
        self.wv_train = conf.data.wv_train
        self.wv_eval = conf.data.wv_eval
        
        self.raw_loader = RawDataLoader(self.conf)
        self.processor = get_processor(self.conf)
        self.index_map = None
        self.dataset = None
        self.train = None
        self.valid = None
        self.test = None
        self.P = None
        self.mean_vec = None
        
        self.initialize_cpus(self.n_cpus)

    def initialize_cpus(self, n_cpus):
        # Ensure the number of CPUs doesn't exceed the system's capacity
        if n_cpus > os.cpu_count():
            n_cpus = 1
        self.n_cpus = n_cpus 
        logging.debug("NF_DataModule | Setting CPUS to {}".format(self.n_cpus))
        
    def prepare_data(self):
        # nothing to do here because raw data is loaded on demand
        pass

    def setup(self, stage):
        raw_data = self.raw_loader.load(stage if stage is not None else "fit")
        # Process the raw data with the selected processor.
        self.dataset = self.processor.process(raw_data, self.conf)
        # Create an index map for train/validation split.
        self.index_map = self._create_index_map(raw_data)
        
        # Transfer P and mean_vec from raw_loader if they exist
        if hasattr(self.raw_loader, 'P'):
            self.P = self.raw_loader.P
        if hasattr(self.raw_loader, 'mean_vec'):
            self.mean_vec = self.raw_loader.mean_vec
            
        if stage in ["fit", None]:
            self.setup_train_val()
        elif stage == "test":
            self.setup_train_val()

    def _create_index_map(self, data):
        index_map = {'train': [], 'valid': []}
        for i in range(len(data['tag'])):
            key = 'valid' if data['tag'][i] == 0 else 'train'
            index_map[key].append(i)
        return index_map
                
    def get_datapath(self, wv_idx=None):
        """Based on params, return the correct dataset we'll be using"""
        if not self.conf.data.buffer:
            return os.path.join(self.path_data, 'preprocessed_data', 'dataset_nobuffer.pt')
        elif self.conf.model.arch == 'modelstm':
            return os.path.join(self.path_data, 'preprocessed_data', f"dataset_{self.conf.model.modelstm.method}.pt")
        else:
            if wv_idx is None:
                raise ValueError("Wavelength index is required for dataset retrieval")
            wv = str(self.wv_dict[wv_idx]).replace('.', '')
            return os.path.join(self.path_data, 'preprocessed_data', f'dataset_{wv}.pt')
        
    def update_near_fields(self, mlp_predictions):
        """Update the near fields with MLP predictions for pipeline evaluation"""
        # Get raw data from the loader
        raw_data = self.raw_loader.load('test')
        
        # Update the near fields with MLP predictions only for validation set
        # Note: We're updating channel 0 of the time dimension (initial condition)
        raw_data['near_fields'][self.index_map['valid'], :, :, :, 0] = mlp_predictions['valid']
            
        # Reprocess the data with updated near fields
        self.dataset = self.processor.process(raw_data, self.conf)
        
        # Reset the train/val splits
        self.setup_train_val()
        
    def setup_train_val(self):
        self.train = Subset(self.dataset, self.index_map['train'])
        self.valid = Subset(self.dataset, self.index_map['valid'])

    #def setup_test(self): #TODO: redundant?
    #    self.test_subset = self.valid_subset

    def setup_fold(self, train_idx, val_idx):
        self.train = Subset(self.dataset, train_idx)
        self.valid = Subset(self.dataset, val_idx)

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          shuffle=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          shuffle=False,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.n_cpus,
                          shuffle=False,
                          persistent_workers=True)
        
def select_data(conf):
    return NFDataModule(conf)

# --------------------------------
# Dataset Classes
# --------------------------------

class WaveMLP_Dataset(Dataset):
    """
    Dataset for the MLP models that map design parameters to fields.
    """
    def __init__(self, data, transform, approach=0, patch_size=1, buffer=True):
        logging.debug("datamodule.py - Initializing WaveMLP_Dataset")
        self.transform = transform
        logging.debug("NF_Dataset | Setting transform to {}".format(self.transform))
        self.approach = approach
        self.patch_size = patch_size
        self.data = data
        self.is_buffer = buffer
        self.format_data() # setup data accordingly
        if self.approach == 2: # distributed subset approach
            self.distributed_indices = self.get_distributed_indices()
            
    def get_distributed_indices(self):
        if self.patch_size == 1: # center the single pixel on the middle
            middle_index = 165 // 2
            return np.array([[middle_index, middle_index]])
        else: # generate patch_size evenly distributed indices
            x = np.linspace(0, 165, self.patch_size).astype(int)
            y = np.linspace(0, 165, self.patch_size).astype(int)
            return list(itertools.product(x, y))

    def __len__(self):
        return len(self.near_fields)

    def __getitem__(self, idx):
        near_field = self.near_fields[idx] # [2, 166, 166]
        radius = self.radii[idx].float() # [9]
        
        if self.approach == 2:
            # selecting patch_size evenly distributed pixels
            x_indices, y_indices = zip(*self.distributed_indices)
            logging.debug(f"WaveMLP_Dataset | x_indices: {x_indices}, y_indices: {y_indices}")
            near_field = near_field[:, x_indices, y_indices]
            near_field = near_field.reshape(2, self.patch_size, self.patch_size)
        if self.transform:   
            near_field = self.transform(near_field)
        
        if self.approach == 3:
            # near field is [2,166,166,2], return the separated dim3
            return near_field[:, :, :, 1], near_field[:, :, :, 0]
        else:
            return near_field, radius
    
    def format_data(self):
        self.radii = self.data['radii']
        self.phases = self.data['phases']
        self.derivatives = self.data['derivatives']
        if not self.is_buffer: # old buffer dataset (U-NET data)
            # focus on 1550 wavelength in y for now
            temp_nf_1550 = self.data['all_near_fields']['near_fields_1550']
            temp_nf_1550 = torch.stack(temp_nf_1550, dim=0) # stack all sample tensors
            temp_nf_1550 = temp_nf_1550.squeeze(1) # remove redundant dimension
            temp_nf_1550 = temp_nf_1550[:, 1, :, :, :] # [num_samples, mag/phase, 166, 166]
            # convert to cartesian coords
            mag, phase = mapping.polar_to_cartesian(temp_nf_1550[:, 0, :, :], temp_nf_1550[:, 1, :, :])
            mag = mag.unsqueeze(1)
            phase = phase.unsqueeze(1)
            self.near_fields = torch.cat((mag, phase), dim=1) # [num_samples, r/i, 166, 166]
        else: # using newer dataset (dataset.pt), simulated for time series models
            temp_nf_1550 = self.data['near_fields'] # [num_samples, 2, 166, 166, 63]
            if self.approach == 3:
                self.near_fields = torch.stack((temp_nf_1550[..., 0], temp_nf_1550[..., -1]), dim=4) # [num_samples, 2, 166, 166, 2]
            else:
                # grab the final slice
                self.near_fields = temp_nf_1550[:, :, :, :, 0] # [num_samples, 2, 166, 166]
            
class WaveModel_Dataset(Dataset):
    """
    Dataset for the time series models associated with emulating wave propagation.
    """
    def __init__(self, samples, labels):
        logging.debug("datamodule.py - Initializing WaveModel_Dataset")
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.samples)

# --------------------------------
# Formatting & Processing Functions
# --------------------------------

# for saving preprocessed data into a single pt. file (LSTM/RNN)
def load_pickle_data(train_path, valid_path, save_path, arch='mlp'):
    near_fields = []
    phases = []
    derivatives = []
    radii = []
    tag = []
    
    for path in [train_path, valid_path]:
        # create path if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)
        # keeping track of original split
        normalized_path = os.path.normpath(path)
        parent_dir = os.path.basename(normalized_path)
        is_train = parent_dir == 'train'
        current_tag = 1 if is_train else 0
        
        for current_file in tqdm(os.listdir(path),
                                desc="Compiling data - {}".format(path),
                                ncols=80,
                                file=sys.stdout,
                                mininterval=1.0): # loop through pickle files
            if current_file.endswith(".pkl"):
                current_file_path = os.path.join(path, current_file)
                tag.append(current_tag) # train or valid sample
                
                with open(current_file_path, "rb") as f:
                    data = pickle.load(f)
                    
                    if arch=='mlp':
                        # extracting final slice from meta_atom_rnn data (1550 wl)
                        near_field_sample = data['data'][:, :, :, -1].float()  # [2, 166, 166]
                    elif arch=='lstm':
                        # all slices
                        near_field_sample = data['data'].float()  # [2, 166, 166, 63]
                    else:
                        raise ValueError("Invalid architecture")
                    
                    # append near field and phase data
                    near_fields.append(near_field_sample)
                    phases.append(data['LPA phases'])
                    
                    # per phases, calculate derivatives and append
                    der = curvature.get_der_train(data['LPA phases'].view(1, 3, 3))
                    derivatives.append(der)
                    
                    # per phase, compute radii and store
                    temp_radii = torch.from_numpy(mapping.phase_to_radii(data['LPA phases']))
                    radii.append(temp_radii)
    
    # convert to tensors
    near_fields_tensor = torch.stack([torch.tensor(f) for f in near_fields], dim=0)  # [num_samples, 2, 166, 166, 63]
    phases_tensor = torch.stack([torch.tensor(p) for p in phases], dim=0)  # [num_samples, 9]
    derivatives_tensor = torch.stack([torch.tensor(d) for d in derivatives], dim=0)  # [num_samples, 3, 3]
    radii_tensor = torch.stack([torch.tensor(r) for r in radii], dim=0)  # [num_samples, 9]  
    tag_tensor = torch.tensor(tag) # [num_samples] 1 for train 0 for valid
    
    logging.debug(f"Near fields tensor size: {near_fields_tensor.shape}")
    logging.debug(f"Memory usage: {near_fields_tensor.element_size() * near_fields_tensor.nelement() / 1024**3:.2f} GB")
    
    torch.save({'near_fields': near_fields_tensor, 
                'phases': phases_tensor, 
                'derivatives': derivatives_tensor,
                'radii': radii_tensor,
                'tag': tag_tensor}, save_path)
    print(f"Data saved to {save_path}")
    
def format_temporal_data(data, conf, order=(-1, 0, 1, 2)):
    """Formats the preprocessed data file into the correct setup  
    and order for the LSTM model.

    Args:
        data (tensor): the dataset
        conf (dict): configuration parameters
        order (tuple, optional): order of the sequence to be used. Defaults to (-1, 0, 1, 2).
        
    Returns:
        dataset (WaveModel_Dataset): formatted dataset
    """
    
    all_samples, all_labels = [], []
    spacing_mode = conf.model.spacing_mode
    io_mode = conf.model.io_mode
    seq_len = conf.model.seq_len
    
    fields = data['near_fields']
    
    '''import matplotlib.pyplot as plt
    import datetime
    print(f"Specific raw_data sample from right before formatting:[0, 0, 0:2, 0:2]: {fields[0, 0, 0:2, 0:2, 0]}")
    # create a plot sample in the real
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(fields[0, 0, :, :, 0], cmap='viridis')
    ax.set_title('raw_data real before formatting')
    ax.axis('off')
    plt.show()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(f'raw_data_real_before_formatting_{timestamp}.pdf')'''
    
    #if stage == 'train': # normalize
    #fields = mapping.l2_norm(data['near_fields'])
        
    # [samples, 2, xdim, ydim, 63] --> access each of the datapoints
    for i in range(fields.shape[0]):       
        full_sequence = fields[i] # [2, xdim, ydim, total_slices]

        total = full_sequence.shape[-1] # all time slices
        
        if spacing_mode == 'distributed':
            if io_mode == 'one_to_many':
                # calculate seq_len+1 evenly spaced indices
                indices = np.linspace(1, total-1, seq_len+1)
                distributed_block = full_sequence[:, :, :, indices]
                # the sample is the first one, labels are the rest
                sample = distributed_block[:, :, :, :1]  # [2, xdim, ydim, 1]
                label = distributed_block[:, :, :, 1:]  # [2, xdim, ydim, seq_len]
                
            elif io_mode == 'many_to_many':
                # Calculate seq_len+1 evenly spaced indices for input and shifted output
                indices = np.linspace(0, total-1, seq_len+1).astype(int)
                distributed_block = full_sequence[:, :, :, indices]
                
                # Input sequence: all but last timestep
                sample = distributed_block[:, :, :, :-1]  # [2, xdim, ydim, seq_len]
                # Output sequence: all but first timestep
                label = distributed_block[:, :, :, 1:]   # [2, xdim, ydim, seq_len]
                
            else:
                # many to one, one to one not implemented
                raise NotImplementedError(f'Specified recurrent input-output mode is not implemented.')
            
            # rearrange dims and add to lists
            sample = sample.permute(order) # [1, 2, xdim, ydim]
            label = label.permute(order) # [seq_len, 2, xdim, ydim]
            # select only one channel
            '''if conf.model.arch == 'modelstm':
                sample = sample[:, 0, :, :]
                label = label[:, 0, :, :]'''
            all_samples.append(sample)
            all_labels.append(label)
            
        elif spacing_mode == 'sequential':
            if io_mode == 'one_to_many':
                #for t in range(0, total, conf['seq_len']+1): note: this raise the total number of sample/label pairs
                t = 0
                # check if there are enough timesteps for a full block
                if t + seq_len < total:
                    block = full_sequence[:, :, :, t:t+seq_len + 1]
                    # ex: sample -> t=0 , label -> t=1, t=2, t=3 (if seq_len were 3)
                    sample = block[:, :, :, 0:1]
                    label = block[:, :, :, 1:]
                        
            elif io_mode == 'many_to_many':
                # true many to many
                sample = full_sequence[:, :, :, :seq_len]
                label = full_sequence[:, :, :, 1:seq_len+1]
                
                # this is our 'encoder-decoder' mode - not really realistic here
                '''step_size = 2 * conf['seq_len']
                #for t in range(0, total, step_size):
                t = 0
                # check if there's enough
                if t + step_size <= total:
                    # input is first seq_len steps in the block
                    sample = full_sequence[:, :, :, t:t+conf['seq_len']]
                    # output is next seq_len steps
                    label = full_sequence[:, :, :, t+conf['seq_len']:t+step_size]'''
                
            else:
                raise NotImplementedError(f'Specified recurrent input-output mode is not implemented.')
                
            sample = sample.permute(order)
            label = label.permute(order)
            '''# select only one channel
            if conf.model.arch == 'modelstm':
                sample = sample[:, 0, :, :]
                label = label[:, 0, :, :]'''
            all_samples.append(sample)
            all_labels.append(label)
        
        else:
            # no other spacing modes are implemented
            raise NotImplementedError(f'Specified recurrent dataloading confuration is not implemented.')
  
    '''# create a plot of the sample real
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(all_samples[0][0, 0, :, :], cmap='viridis')
    ax.set_title('Sample 0, THE LSTM INPUT, first slice')
    ax.axis('off')
    plt.show()
    
    # save the plots
    # get timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.savefig(f'sample_0_input_{timestamp}.pdf')'''
        
    return WaveModel_Dataset(all_samples, all_labels)

def format_ae_data(data, conf):
    """Formats the preprocessed data file into the correct setup  
    and order for the autoencoder pretraining.

    Args:
        data (tensor): the dataset
        conf (dict): configuration parameters
        
    Returns:
        dataset (WaveModel_Dataset): formatted dataset
    """
    all_samples = []
    
    # 100 samples, 63 slices per sample, 63*100 = 6300 samples/labels to train on
    for i in range(data['near_fields'].shape[0]):
        full_sequence = data['near_fields'][i] # [2, xdim, ydim, total_slices]
        for t in range(full_sequence.shape[-1]):
            sample = full_sequence[:, :, :, t] # [2, xdim, ydim] single sample
            all_samples.append(sample)
            
    # were training on reconstruction, so samples == labels
    return WaveModel_Dataset(all_samples, all_samples)
        
def interpolate_fields(data):
    """Interpolates the fields to a lower resolution. Currently supports 2x downsampling.  

    Args:
        data (dict): dictionary containing the near fields, phases, and radii
        
    Returns:
        dataset (WaveModel_Dataset): formatted dataset
    """
    near_fields = data['all_near_fields']['near_fields_1550']
    near_fields = torch.stack(near_fields, dim=0)
    # y-component, real component -> [samples, r/i, xdim, ydim]
    real_fields = near_fields[:, :, 1, 0, :, :] 
    imag_fields = near_fields[:, :, 1, 1, :, :]
    # interpolate and combine r/i
    real_fields_interp = torch.nn.functional.interpolate(real_fields, scale_factor=0.5, mode='bilinear')
    imag_fields_interp = torch.nn.functional.interpolate(imag_fields, scale_factor=0.5, mode='bilinear')
    near_fields_new = torch.cat((real_fields_interp, imag_fields_interp), dim=1)
    # create a new list to store interpolated tensors
    near_fields_new_list = []
    for i in range(near_fields.shape[0]):
        # match dimensions accordingly to the original data
        modified = torch.zeros(1, 3, 2, 83, 83)
        modified[0, 1, :, :, :] = near_fields_new[i]
        near_fields_new_list.append(modified)
    
    # update the data    
    data['all_near_fields']['near_fields_1550'] = near_fields_new_list
    
    return data

def preprocess_svd_data(data, conf):
    """Formats the preprocessed data file into the correct setup  
    and order for the use in format_temporal_data.

    Args:
        data (dict): the dataset
        conf (dict): configuration parameters
        
    Returns:
        formatted_data (dict): formatted data
    """
    full_svd_params = data['near_fields']
    
    P = modes.select_top_k(full_svd_params['Vh'].permute(1, 0), conf.model.modelstm.k)
    og_datapath = os.path.join(conf.paths.data, 'preprocessed_data', f"dataset_155.pt")
    og_data = torch.load(og_datapath, weights_only=True)
    combined = modes.encode_dataset(og_data['near_fields'], P, full_svd_params['mean_vec'])
    combined = combined.permute(0, 1, 3, 2).unsqueeze(2) # [samples, 2, 1, k, 63]
    
    '''samples = full_svd_params['U'].shape[0]
    slices = full_svd_params['U'].shape[-1]
    xdim = full_svd_params['U'].shape[2]
    ydim = full_svd_params['Vh'].shape[3]
    
    # select top k singular values
    k = conf.model.modelstm.k
    U_k, S_k, Vh_k = modes.select_top_k_svd(full_svd_params, k)
    # U_k: [samples, 2, 166, k, 63]
    # S_k: [samples, 2, k, 63]
    # Vh_k: [samples, 2, k, 166, 63]
    
    # flatten along spatial dims
    U_flat = U_k.reshape(samples, 2, xdim*k, slices)
    S_flat = S_k.reshape(samples, 2, k, slices)
    Vh_flat = Vh_k.reshape(samples, 2, k*ydim, slices)
    
    # concatenate along the third dimension
    #combined = torch.cat((U_flat, S_flat, Vh_flat), dim=2).unsqueeze(2)
    combined = Vh_flat.reshape(samples, 2, xdim, k, slices)
    print(combined.shape)
    # final shape: [samples, 2, 1, (166*k + k + k*166), 63]'''
    
    # normalize the data across the time dimension
    combined = combined / 180 * np.pi
    
    print(f"combined.shape: {combined.shape}")
    
    # update the data
    data['near_fields'] = combined
    data['tag'] = og_data['tag']
    
    return data, P, full_svd_params['mean_vec']
    