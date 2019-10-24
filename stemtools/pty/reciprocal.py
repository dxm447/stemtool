import math
import numba
import numpy as np
import cupy as cp
import warnings
from numba import cuda
from ..pty import pty_utils as pt
import numba.cuda
numba.cuda.select_device(3)

def reciprocal(real_size,real_calibration):
    four_y = np.fft.fftshift(np.fft.fftfreq(real_size[0], real_calibration))
    four_x = np.fft.fftshift(np.fft.fftfreq(real_size[1], real_calibration))
    Four_Y,Four_X = np.mgrid[four_y,four_x]
    Fourier = np.sqrt((Four_Y ** 2) + (Four_X**2))
    Phi_Wp = np.arccos(Four_x/Fourier)
    Phi_wp[Fourier==0] = 0
    Phi_wp[Four_Y > 0] = (2*np.pi) - Phi_wp[Four_Y > 0]
    D_plus_matrix = np.zeros((real_size[0],real_size[1],real_size[0],real_size[1]),dtype=np.float32)
    D_minu_matrix = np.zeros_like(D_plus_matrix)
    D_zero_matrix = np.zeros_like(D_plus_matrix)
    for ii in arange(real_size[0]):
        for jj in arange(real_size[1]):
            D_plus_matrix[ii,jj,:,:] = (((Four_X + Four_x[ii,jj]) ** 2) + ((Four_Y + Four_Y[ii,jj]) ** 2))) ** 0.5  
            D_minu_matrix[ii,jj,:,:] = (((Four_X - Four_x[ii,jj]) ** 2) + ((Four_Y - Four_Y[ii,jj]) ** 2))) ** 0.5
            D_zero_matrix[ii,jj,:,:] = Fourier
    return D_plus_matrix,D_minus_matrix,D_zero_matrix
