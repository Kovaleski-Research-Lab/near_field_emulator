import os
import numpy as np
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