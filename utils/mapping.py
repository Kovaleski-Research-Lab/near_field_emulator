import os
import numpy as np
import torch
import scipy
from scipy import interpolate
from scipy.interpolate import lagrange
from scipy.interpolate import BSpline
from numpy.polynomial.polynomial import Polynomial

# phase_to_radii() and radii_to_phase() return numpy arrays with float64 values. 
# output shape = (n,)

def get_mapping(which):

    radii = [0.075, 0.0875, 0.1, 0.1125, 0.125, 0.1375, 0.15, 0.1625, 0.175, 0.1875, 0.2, 0.2125, 0.225, 0.2375, 0.25]
    phase_list = [-3.00185845, -2.89738421, -2.7389328, -2.54946247, -2.26906522, -1.89738599, -1.38868364, -0.78489682, -0.05167712, 0.63232107, 1.22268106, 1.6775137, 2.04169308, 2.34964137, 2.67187105]

    radii = np.asarray(radii)
    phase_list = np.asarray(phase_list)

    if(which=="to_phase"):
        tck = interpolate.splrep(radii, phase_list, s=0, k=3)
    
    elif(which=="to_rad"):
        tck = interpolate.splrep(phase_list, radii, s=0, k=3)

    return tck 

def phase_to_radii(phase_list):
    
    mapper = get_mapping("to_rad")
    to_radii = []
    for phase in phase_list:
        to_radii.append(interpolate.splev(phase_list,mapper))

    return np.asarray(to_radii[0])   

def radii_to_phase(radii):
    
    mapper = get_mapping("to_phase")
    to_phase = []
    for radius in radii:    
        to_phase.append(interpolate.splev(radii,mapper))

    return np.asarray(to_phase[0])

def cartesian_to_polar(real, imag):
    """
    Convert cartesian fields to polar fields
    
    Returns:
    - mag: Magnitude of the field
    - phase: Phase of the field
    """
    complex = torch.complex(real, imag)
    mag = torch.abs(complex)
    phase = torch.angle(complex)
    return mag, phase

def polar_to_cartesian(mag, phase):
    """
    Convert polar fields to cartesian fields
    
    Returns:
    - real: Real part of the field
    - imag: Imaginary part of the field
    """
    complex = mag * torch.cos(phase) + 1j * mag * torch.sin(phase)
    # separate into real and imaginary
    real = torch.real(complex)
    imag = torch.imag(complex)
    return real, imag

def to_plain_dict(obj):
    """Recursively convert Python objects with __dict__ into plain dictionaries."""
    """For saving Pydantic config back to a base YAML file"""
    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    elif hasattr(obj, '__dict__'):
        return {k: to_plain_dict(v) for k, v in vars(obj).items()}
    elif isinstance(obj, list):
        return [to_plain_dict(v) for v in obj]
    else:
        return obj
    
def l1_norm(data):
    """Assuming data of shape [samples, channels, H, W, slices]"""
    sums = data.sum(dim=(1,4), keepdim=True)
    sums = sums + 1e-8 # avoid killing puppies
    data = data / sums
    return data, sums

def l2_norm(data):
    """Assuming data of shape [samples, channels, H, W, slices]"""
    sums_of_squares = (data**2).sum(dim=(1,4), keepdim=True)
    l2_norms = sums_of_squares.sqrt()
    l2_norms = l2_norms + 1e-8
    data = data / l2_norms
    return data, l2_norms

def standardize(data):
    """Assuming data of shape [samples, channels, H, W, slices]"""
    means = data.mean(dim=(0,2,3,4), keepdim=True)
    stds = data.std(dim=(0,2,3,4), keepdim=True)
    data = (data - means) / stds
    return data, means, stds

def sync_power(data):
    # Calculate magnitude for each position
    real = data[:, 0, :, :, :]  # [samples, H, W, slices]
    imag = data[:, 1, :, :, :]  # [samples, H, W, slices]
    magnitude_squared = real**2 + imag**2  # [samples, H, W, slices]
    
    # Calculate total power for each slice
    power_per_slice = magnitude_squared.sum(dim=(1, 2))  # [samples, slices]
    
    # Calculate mean power across all slices for each sample
    mean_power = power_per_slice.mean()#dim=1, keepdim=True)  # [samples, 1]
    
    # Calculate scaling factor for each slice
    # sqrt because we'll apply this to the field, not the power
    scale_factor = torch.sqrt(mean_power / power_per_slice)  # [samples, slices]
    
    # Add dimensions to match broadcasting
    scale_factor = scale_factor.unsqueeze(1).unsqueeze(2)  # [samples, 1, 1, slices]
    
    # Apply scaling to both real and imaginary components
    normalized = data.clone()
    normalized[:, 0, :, :, :] = real * scale_factor
    normalized[:, 1, :, :, :] = imag * scale_factor
    
    return normalized

def reduce_data_resolution(data: dict) -> dict:
    """
    Reduces the resolution of near fields and radii data while preserving core information.
    Also scales the radii values to be in the range [2.0, 4.0].
    
    Parameters
    ----------
    data: dict
        Dictionary containing 'near_fields' and 'radii' tensors
        near_fields shape: [100, 2, 166, 166, 63]
        radii shape: [100, 9]
        
    Returns
    -------
    dict
        Dictionary with reduced resolution data
        near_fields shape: [100, 2, 56, 56, 63]
        radii shape: [100, 4] with values scaled to [2.0, 4.0]
    """
    # Create a copy of the input data to avoid modifying the original
    reduced_data = data.copy()
    
    # Reduce near fields resolution using bilinear interpolation
    near_fields = data['near_fields']  # [100, 2, 166, 166, 63]
    reduced_near_fields = torch.nn.functional.interpolate(
        near_fields.reshape(-1, 2, 166, 166),  # Reshape to [100*63, 2, 166, 166]
        size=(56, 56),
        mode='bilinear',
        align_corners=False
    ).reshape(100, 2, 56, 56, 63)  # Reshape back to [100, 2, 56, 56, 63]
    
    # Reduce radii dimensionality using PCA-like approach
    # We'll select the most significant radii that capture the core information
    radii = data['radii']  # [100, 9]
    
    # Calculate the mean radius for each sample
    mean_radius = radii.mean(dim=1, keepdim=True)  # [100, 1]
    
    # Select the 3 most significant radii (excluding the mean)
    # We'll use the radii at positions 0, 4, and 8 to capture the full range
    selected_radii = torch.stack([
        radii[:, 0],  # First radius
        radii[:, 4],  # Middle radius
        radii[:, 8],  # Last radius
        mean_radius.squeeze(1)  # Mean radius
    ], dim=1)  # [100, 4]
    
    # Scale the radii to the range [2.0, 4.0]
    min_val = selected_radii.min()
    max_val = selected_radii.max()
    scaled_radii = 2.0 + (selected_radii - min_val) * (2.0 / (max_val - min_val))
    
    # Update the data dictionary
    reduced_data['near_fields'] = reduced_near_fields
    reduced_data['radii'] = scaled_radii
    
    return reduced_data

def save_reduced_data(data: dict, output_path: str) -> None:
    """
    Saves the reduced resolution data to a .pt file.
    
    Parameters
    ----------
    data: dict
        Dictionary containing the reduced resolution data
    output_path: str
        Path where the .pt file should be saved
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the data
    torch.save(data, output_path)
    print(f"Reduced data saved to {output_path}")