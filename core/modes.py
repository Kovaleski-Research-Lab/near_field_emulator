import torch
import math
from tqdm import tqdm
from scipy.special import genlaguerre
import os
import matplotlib.pyplot as plt
    
'''def svd(x, config):
    """
    Computes the Singular Value Decomposition on the entire sequence for each sample.
    
    Args:
        x (torch.Tensor): Full dataset tensor of size [samples, r/i, xdim, ydim, slices]
        config: Configuration parameters
    
    Returns:
        torch.Tensor: SVD-transformed tensor of size [samples, r_i, 1, k, slices]
    """
    samples, r_i, xdim, ydim, slices = x.size()
    full_svd_params = []
    
    # Iterate over each sample
    for i in tqdm(range(samples), desc='Processing Samples'):
        # Initialize a list to store real and imaginary channels
        channels = []
        
        # Iterate over each channel (Real and Imaginary)
        for c in range(r_i):
            # Extract the data for the current channel and reshape it
            # Original shape: [xdim, ydim, slices]
            # Reshaped to: [xdim * ydim, slices]
            channel_data = x[i, c].reshape(xdim * ydim, slices)  # Shape: [166*166, 63]
            channels.append(channel_data)
        
        # Concatenate both channels along the first dimension
        # Resulting shape: [2 * 166 * 166, 63]
        aggregated_matrix = torch.cat(channels, dim=0)  # Shape: [2*166*166, 63]
        
        # Perform SVD on the aggregated matrix
        # U: [2*166*166, 63], S: [63], Vh: [63, 63]
        try:
            u, s, vh = torch.linalg.svd(aggregated_matrix, full_matrices=False)
        except RuntimeError as e:
            print(f"SVD did not converge for sample {i}. Error: {e}")
            # Handle SVD convergence issues, e.g., assign zeros or skip
            continue
        
        # Store full decomposition for potential full reconstruction later
        full_svd_params.append({
            'u': u,      # [2*xdim*ydim, slices]
            's': s,      # [slices]
            'vh': vh     # [slices, slices]
        })
        
        if i == 0:
            # Find the optimal k that captures 95% of the energy
            k = find_optimal_k_svd(s, threshold=0.95)
            print(f"Optimal k was found to be: {k}")
            # Initialize the output tensor with the desired shape
            # [samples, r_i, 1, k, slices]
            x_svd = torch.zeros(
                samples, r_i, 1, k, slices, 
                device=x.device, dtype=x.dtype
            )
            
        # Extract the top k right singular vectors
        # vh has shape [63, 63], so vh[:k, :] has shape [k, 63]
        topk_v = vh[:k, :]  # Shape: [k, 63]
        
        # Assign the top k singular vectors to each channel and slice
        # This implies that each slice j has a k-dimensional feature vector: topk_v[:, j]
        # We'll broadcast this across both channels
        
        for c in range(r_i):
            # Assign the topk_v for all slices to the output tensor
            # x_svd shape: [samples, r_i, 1, k, slices]
            # topk_v has shape [k, 63]
            # We need to assign [k, slices] to [1, k, slices] for each channel
            x_svd[i, c, 0, :, :] = topk_v  # Broadcasting the same topk_v across channels
    
    return x_svd, full_svd_params'''
    
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
    k_opt = find_optimal_k_svd(S, threshold=0.95)
    
    # 4) store the full SVD params
    full_svd_params = {
        'U': U,
        'S': S,
        'Vh': Vh
    }

    # 5) Return the top-k singular values
    top_k_s = S[:k_opt]  # shape (k,)
    return top_k_s, full_svd_params

def encode_svd(x):
    samples, r_i, xdim, ydim, slices = x.size()
    
    import sys
    sys.stdout.reconfigure(line_buffering=True)

    # 1) Prepare nested lists for storing results
    #    - top_k_s_list[i][c][j] will be a 1D tensor with the top singular values for that slice
    #    - svd_params_list[i][c][j] will be the dictionary of {U, S, Vh} for that slice
    top_k_s_list = [[[None for _ in range(slices)] for _ in range(r_i)] for _ in range(samples)]
    svd_params_list = [[[None for _ in range(slices)] for _ in range(r_i)] for _ in range(samples)]
    
    # We'll also keep a flat list of top_k_s lengths to find min_k across all slices
    all_top_k_lengths = []
    
    # 2) Iterate over each sample, channel, and slice
    for i in tqdm(range(samples), desc='Processing Samples', mininterval=0.1):
        for c in range(r_i):
            for j in range(slices):
                # Extract the 2D matrix [xdim, ydim]
                slice_2d = x[i, c, :, :, j]
                
                # 3) Call the base SVD function
                single_top_k_s, single_svd_params = svd(slice_2d)
                print(f"Single top k s shape: {single_top_k_s.shape}", flush=True)
                top_k_s_list[i][c][j] = single_top_k_s
                svd_params_list[i][c][j] = single_svd_params
                all_top_k_lengths.append(len(single_top_k_s))

    # 4) Find the global minimum k across all slices
    min_k = min(all_top_k_lengths) if len(all_top_k_lengths) > 0 else 0
    print(f"Minimum k across all slices: {min_k}")

    # 5) Now we build a 4D tensor for the top-k singular values: [samples, r_i, min_k, slices]
    #    We'll truncate each slice's singular values to min_k.
    top_k_s = torch.zeros((samples, r_i, min_k, slices), dtype=x.dtype, device=x.device)

    # Fill in that 4D tensor
    for i in range(samples):
        for c in range(r_i):
            for j in range(slices):
                # each top_k_s_list[i][c][j] is a 1D tensor of shape [k_current]
                # we slice out the first min_k entries
                truncated = top_k_s_list[i][c][j][:min_k]
                top_k_s[i, c, :, j] = truncated
                
    # to be respectful of LSTM process, we need a dummy dim 2
    top_k_s = top_k_s.unsqueeze(2) # [samples, r_i, 1, min_k, slices]

    return top_k_s, svd_params_list


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

def reconstruct_svd(top_k_s, svd_params):
    """
    Reconstructs the original data from the SVD decomposition.
    """
    U = svd_params['U']   # [166, 166]
    S = svd_params['S']   # [166]
    Vh = svd_params['Vh'] # [166, 166]

    # Number of components we are reconstructing with
    k = top_k_s.shape[0]

    # U_k: left singular vectors corresponding to top k singular values
    U_k = U[:, :k]  # [166, k]

    # Construct diagonal matrix from top_k_s
    # shape: (k, k)
    S_k = torch.diag(top_k_s)

    # Vh_k: right singular vectors corresponding to top k singular values
    Vh_k = Vh[:k, :] # [k, 166]

    # Approximate reconstruction: M_approx = U_k @ S_k @ Vh_k
    M_approx = U_k @ S_k @ Vh_k  # [166, 166]

    return M_approx

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
        encoded_fields, full_svd_params = encode_svd(near_fields)
        data['full_svd_params'] = full_svd_params
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
    datasets_path = os.path.join(config.paths.data, 'preprocessed_data')
    # grab the original preprocessed data
    full_data = torch.load(os.path.join(datasets_path, f'dataset_155.pt'))
    # encode accordingly
    encoded_data = encode_modes(full_data, config)
    
    # construct appropriate save path
    save_path = os.path.join(datasets_path, f"dataset_{config.model.modelstm.method}.pt")
    if os.path.exists(save_path):
        raise FileExistsError(f"Output file {save_path} already exists!")
    
    # save the new data to disk
    torch.save(encoded_data, save_path)

    
    
    