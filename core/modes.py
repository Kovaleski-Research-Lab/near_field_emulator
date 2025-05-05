import torch
import math
from tqdm import tqdm
from scipy.special import genlaguerre
import os
import matplotlib.pyplot as plt
import sys
import logging
import numpy as np
#sys.stdout.reconfigure(line_buffering=True)
    
#logging.basicConfig(level=logging.DEBUG)

def prepare_training_matrix(data):
    """
    data: shape [samples, r_i, xdim, ydim, slices]
                e.g. [N_samples, channels, 166, 166, T]
    
    Returns: M, shape [N_total, 166*166]
             where N_total = (N_samples * channels * T).
    """
    samples, channels, xdim, ydim, num_slices = data.shape
    
    # Reshape to [N_total, 166, 166], then flatten
    # First, permute so that slices is the second dimension:
    # e.g. [N_samples, channels, T, 166, 166]
    permuted = data.permute(0, 1, 4, 2, 3)
    
    # Now shape is [samples, channels, slices, 166, 166].
    # We'll flatten samples*channels*slices into one dimension
    # => new shape: [N_total, 166, 166]
    reshaped = permuted.reshape(-1, xdim, ydim)  # -1 = samples*channels*slices
    
    # Flatten each slice to [1, 166*166]
    # => shape [N_total, 166*166]
    M = reshaped.view(-1, xdim*ydim)
    
    mean_vec = M.mean(dim=0, keepdim=True)
    M_centered = M - mean_vec
    
    return M_centered, mean_vec

def compute_global_svd(M_centered):
    """
    M_centered: [N_total, 166*166]
    
    Returns:
      U_big: shape [N_total, min(N_total, 27556)]
      S_big: shape [min(N_total, 27556)]
      V_big: shape [27556,  min(N_total, 27556)]
    """
    # Add shape validation
    if len(M_centered.shape) != 2:
        raise ValueError(f"Expected 2D matrix, got shape {M_centered.shape}")
        
    print(f"M_centered shape before SVD: {M_centered.shape}")
    
    # Ensure matrix has valid dimensions for SVD
    if 0 in M_centered.shape:
        raise ValueError(f"Invalid matrix dimensions for SVD: {M_centered.shape}")
        
    # Ensure the matrix is contiguous in memory
    M_centered = M_centered.contiguous()
    
    U_big, S_big, V_big = torch.linalg.svd(M_centered, full_matrices=False)
    
    return U_big, S_big, V_big

def select_top_k(V_big, k):
    """
    V_big: shape [D, N_total] or [D, D] depending on your data
           Typically [27556, R], where R = min(N_total, 27556).
    Returns:
      P: shape [D, k], the top-k principal components
    """
    # The columns of V_big are the right-singular vectors
    # We want the first k columns
    P = V_big[:, :k]   # shape [27556, k]
    return P

def encode_slice(x_2d, P, mean_vec=None):
    """
    x_2d: shape [166, 166]  (or flattened to [1, 27556])
    P   : shape [27556, k]
    mean_vec: shape [27556,] or None (if no mean-centering used)
    
    Returns: a, shape [k,]
    """
    D = 166*166
    # Flatten the 2D slice
    x_flat = x_2d.view(-1)  # shape [27556]
    
    if mean_vec is not None:
        x_flat_centered = x_flat - mean_vec
    else:
        x_flat_centered = x_flat
    
    # a = x^T * P => shape [k]
    # because x_flat_centered is [1, 27556], P is [27556, k]
    a = x_flat_centered @ P  # [k]
    
    return a

def decode_slice(a, P, mean_vec=None):
    """
    a: shape [k,]
    P: shape [27556, k]
    
    Returns: reconstructed 2D slice [166,166]
    """
    x_flat_approx = a @ P.t()  # shape [27556]
    if mean_vec is not None:
        x_flat_approx = x_flat_approx + mean_vec
    
    x_2d_approx = x_flat_approx.view(166, 166)
    return x_2d_approx

def encode_dataset(train_data, P, mean_vec=None):
    """
    train_data: shape [samples, channels, 166, 166, T]
    P         : shape [27556, k]
    mean_vec  : shape [27556, ] or None
    
    Returns:
      a_data: shape [samples, channels, T, k]
         (the encoded latent vectors for each slice)
    """
    samples, channels, xdim, ydim, slices = train_data.shape
    k = P.shape[1]
    a_data = torch.zeros(samples, channels, slices, k, dtype=train_data.dtype)
    
    # We'll do a triple nested loop or vectorized approach
    for i in tqdm(range(samples), desc='Encoding Samples'):
        for c in range(channels):
            for t in range(slices):
                x_2d = train_data[i, c, :, :, t]
                a = encode_slice(x_2d, P, mean_vec)
                a_data[i, c, t] = a
    
    return a_data  # shape [samples, channels, slices, k]

def decode_dataset(a_data, P, mean_vec=None):
    """
    a_data: shape [samples, channels, slices, k]
    P: shape [27556, k]
    mean_vec: shape [27556, ] or None
    """
    samples, channels, slices, k = a_data.shape
    xdim, ydim = 166, 166
    x_data = torch.zeros(samples, channels, slices, xdim, ydim)
    
    for i in tqdm(range(samples), desc='Decoding Samples'):
        for c in range(channels):
            for t in range(slices):
                a = a_data[i, c, t]
                x_2d = decode_slice(a, P, mean_vec)
                x_data[i, c, t] = x_2d
                
    return x_data


    
def svd(field):
    """
    Perform SVD on a [166,166] tensor, find the optimal k that
    captures at least 'threshold' fraction of the energy, and return
    the top-k singular values as a 1D tensor of shape (k,).
    
    Args:
        field (torch.Tensor): Shape [166,166].
        threshold (float): Fraction of energy to retain.
    
    Returns:
        torch.Tensor: The top-k singular values, shape (k,).
    """
    # 1) Ensure the input is [166,166]
    assert field.shape == (166, 166), \
        f"Expected shape (166,166), got {field.shape}"
    
    # 2) SVD
    #    U: [166,166], S: [166], V^T: [166,166]
    U, S, Vh = torch.linalg.svd(field, full_matrices=False)

    # 3) Find optimal k
    #k_opt = find_optimal_k_svd(S, threshold=0.95)
    k_opt = 10 # 3 seems best after analyzing the data
    
    '''# Diagnostic: Check immediate reconstruction
    direct_recon = U @ torch.diag(S) @ Vh
    immediate_error = torch.abs(field - direct_recon).max()
    logging.debug(f"Immediate SVD reconstruction error: {immediate_error}")'''
    
    # 4) store the full SVD params
    full_svd_params = {
        'U': U,
        'S': S,
        'Vh': Vh
    }

    # 5) Return the top-k singular values
    #top_k_s = S[:k_opt]  # shape (k,)
    return full_svd_params

def encode_svd(x):
    samples, r_i, xdim, ydim, slices = x.size()

    # Pre-allocate tensors for SVD parameters
    U_params = torch.zeros(samples, r_i, slices, xdim, xdim)
    S_params = torch.zeros(samples, r_i, slices, xdim)
    Vh_params = torch.zeros(samples, r_i, slices, xdim, xdim)

    k_opt = 10
    
    for i in tqdm(range(samples), desc='Processing Samples', mininterval=0.1):
        for c in range(r_i):
            for j in range(slices):
                slice_2d = x[i, c, :, :, j]
                params = svd(slice_2d)
                
                # Store parameters in tensors
                U_params[i, c, j] = params['U']
                S_params[i, c, j] = params['S']
                Vh_params[i, c, j] = params['Vh']
                
    # permute to adhere to traditional order
    U_params = U_params.permute(0, 1, 3, 4, 2)
    S_params = S_params.permute(0, 1, 3, 2)
    Vh_params = Vh_params.permute(0, 1, 3, 4, 2)
    
    # Package SVD params in a dictionary of tensors
    svd_params = {
        'U': U_params,
        'S': S_params,
        'Vh': Vh_params
    }
    
    return svd_params


# I'm at: need a wrapper function to do this for each channel, slice, sample
# need to unify k across all somehow
# need to also return the full SVD params

def find_optimal_k_svd(s, threshold=0.95):
    """
    Find the smallest k such that the top k singular values
    capture at least `threshold` fraction of the total energy.
    """    
    # Compute the total energy
    total_energy = (s ** 2).sum()  # sum of squares of singular values
    
    # Compute the cumulative energy ratio
    cumulative_energy = torch.cumsum(s**2, dim=0)
    energy_ratio = cumulative_energy / total_energy
    
    # Find the smallest k for which energy_ratio[k-1] >= threshold
    # (k-1 because PyTorch indexing is 0-based)
    ks = torch.where(energy_ratio >= threshold)[0]
    if len(ks) == 0:
        # Means even if we take all singular values, we don't reach threshold
        k_opt = len(s)
    else:
        k_opt = ks[0].item() + 1  # +1 to turn index into count
    
    return k_opt

def reconstruct_svd(svd_params):
    """
    Reconstructs the original data from the SVD decomposition.
    """
    U_k = svd_params['U']   # [166, k]
    S_k = svd_params['S']   # [166]
    Vh_k = svd_params['Vh'] # [166, k]

    # Number of components we are reconstructing with
    #k = top_k_s.shape[-1]
    
    # replace top k singular values with the one's passed in (i.e. the preds)
    #S[:k] = top_k_s

    # U_k: left singular vectors corresponding to top k singular values
    #U_k = U[:, :k]  # [166, k]
    # Construct diagonal matrix from top_k_s
    # shape: (k, k)
    #S_k = torch.diag(top_k_s)
    # Vh_k: right singular vectors corresponding to top k singular values
    #Vh_k = Vh[:k, :] # [k, 166]
    
    # convert to tensors
    S_k = torch.tensor(S_k)
    U_k = torch.tensor(U_k)
    Vh_k = torch.tensor(Vh_k)

    # Approximate reconstruction: M_approx = U_k @ S_k @ Vh_k
    M_approx = U_k @ np.diag(S_k) @ Vh_k
    # same as above but in numpy

    return M_approx

def reconstruct_full_dataset(x_svd, conf):
    samples = x_svd.shape[0]
    r_i = x_svd.shape[2]
    slices = x_svd.shape[1]
    k = conf.model.modelstm.k
    u_size = 166 * k
    vh_size = 166 * k
    x_svd = x_svd.squeeze(3) # [samples, slices, r_i, U + S + Vh]
    reconstructed = torch.zeros(samples, slices, r_i, 166, 166)
    for i in tqdm(range(samples), desc='Processing Samples', mininterval=0.1):
        for c in range(r_i):
            for j in range(slices):
                # x_svd is [samples, slices, r_i, U + S + Vh]
                svd_params = {
                    'U': x_svd[i, j, c, :u_size].reshape(166, k),
                    'S': x_svd[i, j, c, u_size:u_size+k].reshape(k),
                    'Vh': x_svd[i, j, c, u_size+k:u_size+k+vh_size].reshape(k, 166)
                }
                reconstructed[i, j, c, :, :] = reconstruct_svd(svd_params)
                
    return reconstructed.cpu().numpy()

def select_top_k_svd(full_svd_params, k):
    U_full = full_svd_params['U']
    S_full = full_svd_params['S']
    Vh_full = full_svd_params['Vh']
    
    # select top k singular values
    U_k = U_full[:, :, :, :k, :]
    S_k = S_full[:, :, :k, :]
    Vh_k = Vh_full[:, :, :k, :, :]
    
    return U_k, S_k, Vh_k

def random_proj(x, config):
    """
    Computes a random projection/Johnson Lindenstrauss for the data.
    For simplicity, we create a random projection matrix to reduce ydim to k
    
    Args:
        x (tensor): Full dataset tensor of size [samples, r/i, xdim, ydim, slices]
        config: configuration parameters
    """
    # i_dims is our latent dimensionality
    k = config.model.modelstm.i_dims
    samples, r_i, xdim, ydim, slices = x.size()
    d = xdim * ydim
    
    # calculate single dim for output
    k_dim = int(math.sqrt(k))
    # Verify k is a perfect square
    if k_dim * k_dim != k:
        raise ValueError(f"i_dims ({k}) must be a perfect square")
    # init output [samples, r_i, sqrt(k), sqrt(k), slices]
    x_rp = torch.zeros(samples, r_i, k_dim, k_dim, slices, 
                       device=x.device, dtype=x.dtype)
    
    # reproducibility
    torch.manual_seed(config.model.modelstm.seed)
    
    # create a random projection matrix [d, k]
    w = torch.randn(d, k, device=x.device, dtype=x.dtype) / math.sqrt(k)
    
    for i in tqdm(range(samples), desc='Processing Samples'):
        for j in range(slices):
            # extract channels
            real = x[i, 0, :, :, j].reshape(-1) # [d]
            imag = x[i, 1, :, :, j].reshape(-1) # [d]
            
            # apply projection: [d] * [d, k]
            real_emb = real @ w # [k]
            imag_emb = imag @ w # [k]
            
            # reshape embeddings to square matrices
            real_emb = real_emb.reshape(k_dim, k_dim)
            imag_emb = imag_emb.reshape(k_dim, k_dim)
            
            # Store results
            x_rp[i, 0, :, :, j] = real_emb
            x_rp[i, 1, :, :, j] = imag_emb

    return x_rp
    

def generate_gl_modes(xdim, ydim, k, w0, p_max, l_max, device, dtype):
    """
    Generate at least k Gaussian-Laguerre modes on a 2D grid.
    We'll enumerate (p,l) pairs from p=0..p_max, l=-l_max..l_max
    until we have at least k modes.

    Returns:
        modes: complex tensor of shape [k, xdim, ydim], complex64 or complex128
        (Depending on dtype support. We'll use torch.complex64 if dtype is float32.)
    """
    # create coord grid
    # center grid from -1 to 1 in both x and y #TODO real physical scale?
    x_lin = torch.linspace(-1.0, 1.0, xdim, device=device, dtype=dtype)
    y_lin = torch.linspace(-1.0, 1.0, ydim, device=device, dtype=dtype)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing='ij') # [xdim, ydim]
    
    R = torch.sqrt(X**2 + Y**2) # radius
    Theta = torch.atan2(Y, X) # angle
    
    modes_list = []
    count = 0
    
    # enumerate (p, l) pairs
    for p in tqdm(range(p_max+1), desc='Generating GL Modes'):
        for l in range(-l_max, l_max+1):
            if count >= k:
                break
            # Compute LG_{p}^{l}(r, theta)
            # Laguerre polynomial: L_p^{|l|}(x)
            # set x = 2r^2/w0^2
            r_scaled = 2.0 * R.pow(2) / (w0**2)
            L_p_l = torch.from_numpy(genlaguerre(p, abs(l))(r_scaled.cpu().numpy())).to(device=device, dtype=dtype)
            
            # LG mode formula (up to normalization constant)
            # common norm in lieu of exact constants #TODO okay?
            # LG_{p}^{l}(r, theta) ~ (sqrt(2)*r/w0)^{|l|} * exp(-r^2/w0^2) * L_p^{|l|}(2r^2/w0^2) * exp(i l θ)
            # TODO add normalization constant
            radial_part = (math.sqrt(2.0) * R / w0).pow(abs(l)) * torch.exp(-R.pow(2)/ (w0**2)) * L_p_l
            phase_part = torch.exp(1j * l * Theta)  # complex
            mode = radial_part * phase_part  # complex-valued
            
            modes_list.append(mode)
            count += 1
        if count >= k:
            break
        
    # [k, xdim, ydim]
    modes = torch.stack(modes_list, dim=0)
    return modes

def gauss_laguerre_proj(x, config):
    """
    Project the input field onto Gaussian-Laguerre modes.

    Args:
        x (tensor): [samples, r_i, xdim, ydim, slices]
        config: configuration parameters
    """
    k = config.model.modelstm.i_dims
    w0 = config.model.modelstm.w0
    p_max = config.model.modelstm.p_max
    l_max = config.model.modelstm.l_max
    
    samples, r_i, xdim, ydim, slices = x.size()
    
    # generate k modes [k, xdim, ydim] (complex)
    modes = generate_gl_modes(xdim, ydim, k, w0, p_max, l_max, x.device, x.dtype)
    
    # calculate single dim for output
    k_dim = int(math.sqrt(k))
    # Verify k is a perfect square
    if k_dim * k_dim != k:
        raise ValueError(f"i_dims ({k}) must be a perfect square")
    x_lg = torch.zeros(samples, r_i, k_dim, k_dim, slices, device=x.device, dtype=x.dtype)
    
    # Projection: coeff: sum over x,y of E(x,y)*conjugate(mode(x,y))
    # E(x,y) = E_r + i E_i. mode is complex
    for i in tqdm(range(samples), desc='Processing Samples'):
        for j in range(slices):
            # construct complex field for this slice
            E_r = x[i, 0, :, :, j]
            E_i = x[i, 1, :, :, j]
            E = torch.complex(E_r, E_i)
            
            # compute inner products
            modes_conj = torch.conj(modes) # [k, xdim, ydim]
            E_expanded = E.unsqueeze(0) # Expanded: [1, xdim, ydim], broadcast mul
            coeffs = (E_expanded * modes_conj).sum(dim=(-2, -1)) # [k]
            
            #print(f'real coeffs shape: {coeffs.real.shape}')
            
            # reshape embeddings to square matrices - friendly for datamodule 
            # but we flatten back for actual processing
            real = coeffs.real.reshape(k_dim, k_dim)
            imag = coeffs.imag.reshape(k_dim, k_dim)
            
            # update output tensor
            x_lg[i, 0, :, :, j] = real
            x_lg[i, 1, :, :, j] = imag
            
    return x_lg
    
def fourier_modes(x, config):
    """
    Encoding fourier modes on the input data
    
    Args:
        x (tensor): Full dataset tensor of size [samples, r/i, xdim, ydim, slices]
        config: configuration parameters
    """
    pass

def encode_modes(data, config):
    """Takes the input formatted dataset and applies a specified modal decomposition

    Args:
        data (tensor): the dataset
        config: mode encoding parameters
        
    Returns:
        dataset (WaveModel_Dataset): formatted dataset with encoded data
    """
    near_fields = data['near_fields'].clone()
    
    method = config.model.modelstm.method
    
    if method == 'svd': # encoding singular value decomposition
        #encoded_fields = encode_svd(near_fields)
        # Ensure near_fields is properly shaped before centering
        if len(near_fields.shape) != 5:  # [samples, channels, H, W, slices]
            raise ValueError(f"Expected 5D tensor, got shape {near_fields.shape}")
        
        M_centered, mean_vec = prepare_training_matrix(near_fields)
        U_big, S_big, V_big = compute_global_svd(M_centered)
        encoded_fields = {
            'mean_vec': mean_vec,
            'U': U_big,
            'S': S_big,
            'Vh': V_big
        }
    elif method == 'random': # random projection / Johnson-Lindenstrauss
        encoded_fields = random_proj(near_fields, config)
    elif method == 'gauss': # gauss-laguerre modes
        encoded_fields = gauss_laguerre_proj(near_fields, config)
    elif method == 'fourier': # a fourier encoding
        encoded_fields = fourier_modes(near_fields, config)
    else:
        raise NotImplementedError(f"Mode encoding method '{method}' not recognized.")
    
    # update the real data
    data['near_fields'] = encoded_fields
    
    return data

def run(config):
    datasets_path = os.path.join(config.paths.data)
    if config.directive == 7: # encoding
        # grab the original preprocessed data
        full_data = torch.load(os.path.join(datasets_path, f'dataset_155.pt'), weights_only=True)

        mask = full_data['tag'] == 1
        print(f"Number of samples after filtering: {mask.sum()}")

        
        filtered_data = {
            'near_fields': full_data['near_fields'][mask],
            'tag': full_data['tag'][mask]
        }
        print(f"Filtered data shape before random selection: {filtered_data['near_fields'].shape}")
        
        # randomly select half of the data
        shrink = int(filtered_data['near_fields'].shape[0] / 10)
        filtered_data['near_fields'] = filtered_data['near_fields'][:shrink]
        filtered_data['tag'] = filtered_data['tag'][:shrink]
        print(f"Filtered data shape after random selection: {filtered_data['near_fields'].shape}")

        # encode accordingly
        encoded_data = encode_modes(filtered_data, config)
            
        # construct appropriate save path
        save_path = os.path.join(datasets_path, f"dataset_{config.model.modelstm.method}.pt")
        if os.path.exists(save_path):
            raise FileExistsError(f"Output file {save_path} already exists!")
        
        # save the new data to disk
        torch.save(encoded_data, save_path)
    else: # directive == 8 and we're decoding #TODO right now this only works for SVD
        encoded_data = torch.load(os.path.join(datasets_path, f'dataset_{config.model.modelstm.method}.pt'))
        svd_params = torch.load(os.path.join(datasets_path, f'svd_params.pt'))
        

    
    
    