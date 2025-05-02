# Near Field Wave Emulation

Exploration of deep learning models for emulating wavefront propagation and response from metasurfaces. Supports two distinct problems: Time-series networks for modeling propagation directly, and MLP-based methods for mapping metasurface parameters to downstream fields. Currently supports the following architectures: MLP, CVNN, LSTM, Autoencoder-LSTM, Autoencoder-ConvLSTM, LSTM with mode encoding, Transformer, and Inverse Design (naive, tandem).

## Description of Contents

- `kube/` : contains files for loading, configuring, and launcing Kubernetes jobs. This includes scripts for training, evaluation, and analyzing results.
  - **launch_training.py** : This file generates a useable YAML file for a training instance; similar files exist for evaluation and data manipulation
  - **save_eval_plots.py** : necessary helper for saving evaluation plots from cloud storage to disk for viewing
- `core/` : contains various key files.
  - `models/` : contains model implementations
    - **WaveMLP.py** : MLP implementation for metasurface -> field
    - **WaveModel.py** : Base abstract class for wavefront propagation problem
    - **WaveXXXX.py** : Subclass implementation of model XXXX
    - **ConvLSTM.py** : Convolutional LSTM implementation
    - **CVNN.py** : implementation of Complex-Valued NN, activations
    - **autoencoder.py** : autoencoder implementation
  - **datamodule.py** : Master file handling dataloading for both problems
  - **modes.py** : Defines different mode encoding approaches (SVD, Random Projection, etc.)
  - **preprocess_data.py** : Handles the formatting of data .pkl files for time series networks, splitting into train/valid.
  - **compile_data.py** : Data after preprocessing is saved as .pkl files, This file contains a function that compiles it into a single pytorch file for use with the model.
  - **train.py** : The training process 
- `evaluation/` : contains files used for eval pipeline
  - **eval_model.py** : Primary file for evaluation process
  - **evaluation.py** : File containing all plotting, measuring, etc. methods
  - **inference.py** : Contains additional evaluation functions for real-time analysis
- `utils/` : Helper files
  - **mapping.py** : Primarily contains the functions `cartesian_to_polar` and its vice versa counterpart for converting between polar and cartesian before displaying plots
  - **model_loader.py** : This wrapper dynamically determines which model we're using
  - **parameter_manager.py** : Formats and organizes `config.yaml` contents in a manner consistent with the needs of files which reference them such as the model, data module, etc.
  - **visualize.py** : contains implementation of plotting function that shows animation of field progression - can be seen in the presentation
- `build/` : Contains the Dockerfile for building the Docker container.
- `conf/config.yaml` : Specifies all aspects of training, evaluation, setup, everything.
- `main.py` : The driver file, refers to `config.yaml` to determine what to do when a process is started.

## Configuration

The `config.yaml` file controls all aspects of training and evaluation. Key parameters include:

- `directive`: 
  - `0`: Train network
  - `1`: Run evaluation
  - `2`: Data compilation
  - `3`: Load results
  - `4`: Preprocess and format data
  - `5`: Perform mode encoding on data (modeLSTM)

- `deployment`: 
  - `0`: Local deployment
  - `1`: Kubernetes deployment

- `arch`:
  - `0`: Dual MLPs (separate real/imaginary)
  - `1`: CVNN (complex-valued)
  - `2`: LSTM
  - `3`: ConvLSTM
  - `4`: Autoencoder-Wrapped LSTM
  - `5`: Autoencoder-Wrapped ConvLSTM
  - `6`: LSTM with specific modes encoded first
  - `7`: img2video diffuser model (In Progress)
  - `8`: autoencoder

- `model_id`: Unique identifier for the model
- `batch_size`: Training batch size
- `num_epochs`: Maximum training epochs
- `learning_rate`: Initial learning rate
- `cross_validation`: whether to do cross validation or not
- `kube`: specifies parameters used to create kubernetes YAMLs
- additional configurations for architecture, training, etc.
- these addl parameters are explained where necessary within the file itself

## Running the Code

### Prerequisites

1. Ensure you have access to Dr. K's Lab Org [Kovaleski-Research-Lab](https://github.com/Kovaleski-Research-Lab). Certain companion intstructions require you have access to view setup steps/passwords/etc.
2. Having the following directory structure on your local machine will minimize the potential for errors but it isn't strictly necessary:

```
develop/  
│
└───code/
│   │
│   └───(pull down this repo here)
│   
└───data/
│   │ 
│   └───preprocessed_data/
│
└───results/
```

3. Install Kubernetes: [Kube Setup Process](https://github.com/Kovaleski-Research-Lab/Global-Lab-Repo/blob/main/sops/software_development/kubernetes.md)
4. Setup Docker: [Docker Setup](https://github.com/Kovaleski-Research-Lab/Global-Lab-Repo/blob/main/sops/software_development/docker.md)
5. Configure SSH **deploy key** authentication: [Setting up Deploy Key](https://github.com/Kovaleski-Research-Lab/Global-Lab-Repo/blob/main/sops/software_development/github-deploy-key.md)

### Docker

A docker container must be created to do any kind of running with this code. To build, execute the following when starting from the root directory:

```
cd build
docker build -t kovaleskilab/ml_basic:v4 .
```

(Note): Based on the current contents of the `Dockerfile` this assumes you have already followed instructions in Prerequisites Step 2 and pulled the associated docker image from dockerhub:

Local Deployment
```
sudo docker pull kovaleskilab/ml_basic:v4
```

Kube Deployment
```
sudo docker pull kovaleskilab/ml_basic:v4-kube
```

From there, we want to run the container but its critical that we mount it utilizing the following scheme to ensure the code exists within the container (if you set up the directory structure introduced earlier, this step is intuitive):

```
sudo docker run \
-v /path/to/data:/develop/data \
-v /path/to/results:/develop/results \
-v {parent directory containing near_field_emulator}:/develop/code \
-v ~/.kube:/root/.kube
  kovaleskilab/ml_basic:v4-kube'
```

Note: For local deployment, just omit the `kube` part and use the other container.

### Basic Usage

From within the Docker container after launching (should be at **/develop/code**):

```
cd near_field_emulator
python3 main.py --config conf/config.yaml
```

### Training

1. Set `directive: 0` in `conf/config.yaml`
2. Choose architecture with `arch` parameter
3. Set desired `model_id` and hyperparameters
4. Run `python3 main.py --config conf/config.yaml`

### Evaluation

1. Set `directive: 1` in `conf/config.yaml`
2. Set other desired parameters
3. Run `python3 main.py --config conf/config.yaml`

Evaluation outputs (saved to `develop/results`, copied to `training_results` PVC):
- Predicted vs actual field distributions - magnitude and phase
- Error metrics (MSE, PSNR, SSIM) - resub and validation
- Flipbooks of wave propagation (for time-series models)

### Loading Results

(Note: Process not fully integrated yet)

1. Set `directive: 2` in `conf/config.yaml`
2. Set `arch` to `convlstm` (for example) to get all eval results for that model
3. Run `python3 main.py --config conf/config.yaml`

### Running Meep Simulations

To generate training data using Meep:

1. Follow instructions in this repo (ensure you are in `nfe_branch`): [general_3x3](https://github.com/Kovaleski-Research-Lab/general_3x3/tree/nfe_branch?tab=readme-ov-file)

### Data Preprocessing

1. Prerequisite: MEEP Simulation(s) have been ran and reduced to volumes in `data/nfe-data/volumes`
2. Copy `neighbors_library_allrandom.pkl` to `/develop/code/near_field_emulator/utils/` if not done already (can copy from `general_3x3` repo)
3. Set `directive: 4` in `conf/config.yaml`
4. Set `deployment` accordingly for deployment type
5. Run `python3 main.py --config conf/config.yaml`

Preprocessed datasets contain:
- Normalized field components
- Phase information
- Derivative calculations

## Kubernetes Jobs

The `kube/` directory contains job templates for distributed training. Creating templates is as simple as setting parameters in `conf/config.yaml` and running the following:

```
python3 -m kube.launch_training # (or launch_evaluation)
```

After the above is ran a complete YAML configuration file is generated and placed in `kube/jobs`. Starting a training job, or evaluation job, or whatever process we're doing then requires running the following from a terminal (example is for convlstm training):

```
kubectl apply -f kube/kube_jobs/convlstm-training.yaml
```

**Note:** Running these jobs requires access to specific Kubernetes namespaces and PVCs, as well as the relevant private github repos. Instructions for setting up required permissions are detailed in
the Prerequisites section above.