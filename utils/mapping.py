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
    sums = data.sum(dim=(4), keepdim=True)
    sums = sums + 1e-8 # avoid killing puppies
    data = data / sums
    return data

def l2_norm(data):
    """Assuming data of shape [samples, channels, H, W, slices]"""
    sums_of_squares = (data**2).sum(dim=(1,2,3), keepdim=True)
    l2_norms = sums_of_squares.sqrt()
    l2_norms = l2_norms + 1e-8
    data = data / l2_norms
    return data