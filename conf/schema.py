from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, field_validator, model_validator, ValidationError
import os
import yaml
#--------------------------------
#       Config Schema
#--------------------------------

class LSTMConfig(BaseModel):
    i_dims: int
    h_dims: int
    num_layers: int

class ConvLSTMConfig(BaseModel):
    num_layers: int
    in_channels: int
    out_channels: int
    kernel_size: int
    padding: int
    spatial: int

class AutoencoderConfig(BaseModel):
    encoder_channels: List[int]
    decoder_channels: List[int]
    latent_dim: int
    pretrained: bool
    freeze_weights: bool
    use_decoder: bool
    spatial: int
    method: Literal['linear', 'conv'] = 'linear'

class ModesConfig(BaseModel):
    num_layers: int
    i_dims: int
    h_dims: int
    spatial: int
    w0: float
    p_max: int
    l_max: int
    seed: int
    method: Literal['svd', 'random', 'gauss', 'fourier'] = 'svd'
    
class DiffusionConfig(BaseModel):
    num_generated_frames: int = 14
    prompt: str = ''
    use_half_precision: bool = True
class ModelConfig(BaseModel):
    arch: str # an int in config.yaml
    model_id: str
    optimizer: str
    learning_rate: float = 1e-3
    lr_scheduler: Literal['CosineAnnealingLR', 'ReduceLROnPlateau']
    num_epochs: int = 0 
    objective_function: str = "mse"
    mcl_params: Dict[str, Any]
    mlp_real: Dict[str, Any]
    mlp_imag: Dict[str, Any]
    cvnn: Dict[str, Any]
    mlp_strategy: int = 0
    patch_size: int = 3
    num_design_conf: int = 9
    interpolate_fields: bool = False
    lstm: LSTMConfig = None
    modelstm: ModesConfig = None
    convlstm: ConvLSTMConfig = None
    autoencoder: AutoencoderConfig = None
    diffusion: DiffusionConfig = None
    seq_len: int = 10
    io_mode: str = "one_to_many"
    autoreg: bool = True
    spacing_mode: str = "sequential"
    
    @field_validator("arch", mode="before")
    def validate_arch(cls, value):
        if isinstance(value, int):
            return get_model_type(value)
        raise ValueError("arch must be an integer between 0 and 8")
    
class TrainerConfig(BaseModel):
    batch_size: int
    num_epochs: int = 100
    accelerator: Literal['cpu', 'gpu'] = 'gpu'
    valid_rate: int = 1
    gpu_config: List[Any] = [True, [0]]
    include_testing: bool = False
    cross_validation: bool = True
    patience: int = 15
    min_delta: float = 0.0001
    load_checkpoint: bool = False
    
class PathsConfig(BaseModel):
    root: str
    data: str
    train: str
    valid: str
    results: str
    volumes: str
    library: str
    pretrained_ae: str
    
    @model_validator(mode="after")
    def validate_paths(cls, model):
        model.root = os.path.abspath(model.root)
        model.data = os.path.join(model.root, model.data)
        model.train = os.path.join(model.data, model.train)
        model.valid = os.path.join(model.data, model.valid)
        model.results = os.path.join(model.root, model.results)
        model.volumes = os.path.join(model.data, model.volumes)
        model.library = os.path.join(model.root, model.library)
        model.pretrained_ae = os.path.join(model.results, model.pretrained_ae)
        return model
    
    @model_validator(mode="after")
    def validate_existence(cls, model):
        if not os.path.exists(model.root):
            raise ValueError(f"Root directory {model.root} does not exist")
        if not os.path.exists(model.data):
            raise ValueError(f"Data directory {model.data} does not exist")
        '''if not os.path.exists(model.train):
            raise ValueError(f"Train directory {model.train} does not exist")
        if not os.path.exists(model.valid):
            raise ValueError(f"Valid directory {model.valid} does not exist")
        if not os.path.exists(model.results):
            raise ValueError(f"Results directory {model.results} does not exist")
        if not os.path.exists(model.volumes):
            raise ValueError(f"Volumes directory {model.volumes} does not exist")
        if not os.path.exists(model.library):
            raise ValueError(f"Library file {model.library} does not exist")
        if not os.path.exists(model.pretrained_ae):
            raise ValueError(f"Pretrained AE directory {model.pretrained_ae} does not exist")'''
        return model
    
class DataConfig(BaseModel):
    n_cpus: int
    n_folds: int
    buffer: bool = True
    wv_dict: Dict[str, float]
    wv_preprocess: str
    wv_train: str
    wv_eval: str
    
    @field_validator("wv_preprocess", mode="before")
    def validate_pp_wavelength(cls, value):
        if len(value) != 1:
            raise ValueError(f"Wavelength index must be a single character")
        possibilities = ['1', '2', '3', '4', '5']
        if value not in possibilities:
            raise ValueError(f"Wavelength index must be one of {possibilities}")
        return value
    
    @field_validator("wv_train", mode="before")
    def validate_train_wavelength(cls, value):
        if len(value) > 5 or len(value) < 1:
            raise ValueError(f"Wavelength must be a string of length 1-5")
        for char in value:
            if char not in ['1', '2', '3', '4', '5']:
                raise ValueError(f"Possible indices are 1, 2, 3, 4, 5")
        return value
    
    @field_validator("wv_eval", mode="before")
    def validate_eval_wavelength(cls, value):
        if len(value) > 5 or len(value) < 1:
            raise ValueError(f"Wavelength must be a string of length 1-5")
        for char in value:
            if char not in ['1', '2', '3', '4', '5']:
                raise ValueError(f"Possible indices are 1, 2, 3, 4, 5")
        return value
    
class PhysicsConfig(BaseModel):
    Nx_metaAtom: int
    Ny_metaAtom: int
    Lx_metaAtom: float
    Ly_metaAtom: float
    n_fusedSilica: float
    n_PDMS: float
    n_amorphousSilica: float
    h_pillar: float
    thickness_pml: float
    thickness_fusedSilica: float
    thickness_PDMS: float
    wavelength: float
    distance: float
    Nxp: int
    Nyp: int
    Lxp: float
    Lyp: float
    adaptive: bool
    
class KubeConfig(BaseModel):
    namespace: Literal['gpn-mizzou-muem']
    image: str
    job_files: str
    pvc_volumes: str
    pvc_preprocessed: str
    pvc_results: str
    data_job: Dict[str, Any]
    train_job: Dict[str, Any]
    load_results_job: Dict[str, Any]
    evaluation_job: Dict[str, Any]
    
    @model_validator(mode="after")
    def validate_existence(cls, model):
        if not os.path.exists(model.job_files):
            raise ValueError(f"Job files directory {model.job_files} does not exist")
        return model

class MainConfig(BaseModel):
    directive: int
    deployment: int
    paths: PathsConfig
    trainer: TrainerConfig
    model: ModelConfig
    data: DataConfig
    physics: PhysicsConfig
    kube: KubeConfig
    seed: List[Any] = field(default_factory=list)
    
    @model_validator(mode="after")
    def validate_results(cls, main):
        # need specific path for good categorization in results
        if main.model.arch == 'modelstm': # further categorize by mode encoding method
            main.paths.results = os.path.join(main.paths.results, main.model.arch, main.model.modelstm.method, main.model.io_mode, main.model.spacing_mode, f"model_{main.model.model_id}")
        elif main.model.arch == 'mlp' or main.model.arch == 'cvnn':
            main.paths.results = os.path.join(main.paths.results, main.model.arch, f"model_{main.model.model_id}")
        else:
            main.paths.results = os.path.join(main.paths.results, main.model.arch, main.model.io_mode, main.model.spacing_mode, f"model_{main.model.model_id}")
        
        return main
    
    @model_validator(mode="after")
    def validate_memory(cls, main):
        # need more memory when loading a bunch of wavelength datasets
        if len(main.data.wv_train) < 2:
            main.kube.train_job['num_mem_lim'] = '150Gi'
            main.kube.train_job['num_mem_req'] = '150Gi'
        elif 2 < len(main.data.wv_train) < 4:
            main.kube.train_job['num_mem_lim'] = '250Gi'
            main.kube.train_job['num_mem_req'] = '250Gi'
        else:
            main.kube.train_job['num_mem_lim'] = '300Gi'
            main.kube.train_job['num_mem_req'] = '300Gi'
        return main
    
#--------------------------------
#       Helper Functions
#--------------------------------
    
def get_model_type(arch: int) -> str:
    model_types = {
        0: "mlp",
        1: "cvnn",
        2: "lstm",
        3: "convlstm",
        4: "ae-lstm",
        5: "ae-convlstm",
        6: "modelstm",
        7: "diffusion",
        8: "autoencoder"
    }
    return model_types.get(arch, ValueError("Model type not recognized"))

#--------------------------------
#       Load Config
#--------------------------------
def load_config(config_path: str) -> MainConfig:
    """
    Load and validate the configuration from config.yaml.
    """
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    
    try:
        # Parse and validate the configuration
        config = MainConfig(**raw_config)
    except ValidationError as e:
        raise ValueError(f"Config validation failed: {e}") from e

    return config