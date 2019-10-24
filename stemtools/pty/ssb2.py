import numpy as np
import cupy as cp

def phase_cupy(complex_arr):
    return(cp.arctan(cp.imag(complex_arr)/cp.real(complex_arr)))

def ampli_cupy(complex_arr):
    return((((cp.imag(complex_arr) ** 2) + (cp.real(complex_arr) ** 2)) ** 0.5))

def e_lambda(voltage_kV):
    m = 9.109383 * (10 ** (-31))  # mass of an electron
    e = 1.602177 * (10 ** (-19))  # charge of an electron
    c = 299792458  # speed of light
    h = 6.62607 * (10 ** (-34))  # Planck's constant
    voltage = voltage_kV * 1000
    numerator = (h ** 2) * (c ** 2)
    denominator = (e * voltage) * ((2*m*(c ** 2)) + (e * voltage))
    wavelength = (10 ** 12) *((numerator/denominator) ** 0.5) #in angstroms
    return wavelength

def ssb_kernel(processed4D,real_calibration,aperture,voltage):
    data_size = processed4D.shape
    wavelength = e_lambda(voltage)
    cutoff = aperture/wavelength
    four_y = np.fft.fftshift(np.fft.fftfreq(data_size[0], real_calibration))
    four_x = np.fft.fftshift(np.fft.fftfreq(data_size[1], real_calibration))
    Four_Y,Four_X = np.meshgrid(four_y,four_x)
    FourXY = np.sqrt((Four_Y ** 2) + (Four_X**2))
    Left_Lobe = np.zeros(data_size,dtype=processed4D.dtype)
    RightLobe = np.zeros_like(Left_Lobe)
    
    #convert to CuPy arrays
    Four_Y = cp.asarray(Four_Y)
    Four_X = cp.asarray(Four_X)
    FourXY = cp.asarray(FourXY)
    Left_Lobe = cp.asarray(Left_Lobe)
    RightLobe = cp.asarray(RightLobe)
    
    lobe_calc(Left_Lobe,RightLobe,Four_Y,Four_X,FourXY,cutoff)
    
    data_phase = phase_cupy(processed4D)
    data_ampli = ampli_cupy(processed4D)
    left_trotter = cp.multiply(cp.multiply(data_ampli,Left_Lobe),cp.exp((1j)*cp.multiply(data_phase,Left_Lobe)))
    left_image = cp.asnumpy(cp.fft.ifft2(cp.sum(left_trotter,axis=(0,1))))
    righttrotter = cp.multiply(cp.multiply(data_ampli,RightLobe),cp.exp((1j)*cp.multiply(data_phase,RightLobe)))
    rightimage = cp.asnumpy(cp.fft.ifft2(cp.sum(righttrotter,axis=(0,1))))
    
    return left_image,rightimage

def lobe_calc(Left_Lobe,RightLobe,Four_Y,Four_X,FourXY,cutoff):
    lobe_shape = Left_Lobe.shape
    two_dims_rolled = lobe_shape[2]*lobe_shape[3]
    Left_Lobe = cp.reshape(Left_Lobe,(lobe_shape[0],lobe_shape[1],two_dims_rolled))
    RightLobe = cp.reshape(RightLobe,(lobe_shape[0],lobe_shape[1],two_dims_rolled))
    for pp in range(int(two_dims_rolled)):
        (ii,jj) = np.unravel_index(pp, (lobe_shape[2],lobe_shape[3]))
        xq = Four_X[ii,jj]
        yq = Four_Y[ii,jj]
        d_plus = (((Four_X + xq) ** 2) + ((Four_Y + yq) ** 2)) ** 0.5  
        d_minu = (((Four_X - xq) ** 2) + ((Four_Y - yq) ** 2)) ** 0.5
        d_zero = FourXY
            
        ll = Left_Lobe[:,:,pp]
        ll[d_plus < cutoff] = 1
        ll[d_minu > cutoff] = 1
        ll[d_zero < cutoff] = 1
        Left_Lobe[:,:,pp] = ll
            
        rr = RightLobe[:,:,pp]
        rr[d_plus > cutoff] = 1
        rr[d_minu < cutoff] = 1
        rr[d_zero < cutoff] = 1
        RightLobe[:,:,pp] = rr
    Left_Lobe = cp.reshape(Left_Lobe,lobe_shape)
    RightLobe = cp.reshape(RightLobe,lobe_shape)
