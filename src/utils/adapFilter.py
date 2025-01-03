"""
author: ankit anand
created on: 23/12/24
"""

from scipy.signal import convolve
import numpy as np

def applyAdaptationFilter(signal:np.ndarray, signalSr:int, filterLength:int):
	"""
	decs
	----
		given any 1d signal, this computes smoothened derivative of the signal using two gaussian profiles
	args
	----
		signal: np.ndarray
			signal to find derivative of
		signalSr: int
			sampling rate of the signal
		filterLength: int
			length of the filter
	returns
	-------
		der: np.ndarray
			derivative of the signal
		kernel: np.ndarray
			kernel used to compute the derivative, one can plot the kernel to better understand the process or perform parameter tuning appropriately
	examples
	--------
	>>> _, kernel = applyAdaptationFilter(np.zeros(1000), 20, 5)
	
	>>> import matplotlib.pyplot as plt
	
	>>> plt.figure(figsize=(13, 2))
	>>> plt.stem(kernel)
	>>> plt.show()
		
	"""
	tau1 = int(0.4 * signalSr) # standard deviation
	d1 = int(0.1 * signalSr) # distance from the center
	tau2 = int(0.4 * signalSr) # standard deviation
	d2 = int(0.1 * signalSr) # distance from the center
	
	kernel = np.zeros(2 * filterLength)
	t = np.arange(-filterLength, +filterLength+1) 
	kernel = (1/(tau1*np.sqrt(2*np.pi))) * np.exp(-(t-d1)**2/(2*tau1**2)) - (1/(tau2*np.sqrt(2*np.pi))) * np.exp(-(t+d2)**2/(2*tau2**2))
	kernel =  np.exp(-(t-d1)**2/(2*tau1**2)) - np.exp(-(t+d2)**2/(2*tau2**2))
	kernel /= np.sum(np.abs(kernel)) # normalise the kernal
	
	# Apply the biphasic filter using convolution
	der = convolve(signal, kernel[::-1], mode='same') # reversed to perform convolution in the right orientation
	der[der>0] = 0 # for this task we are only interested in drop in intensity
	
	return der, kernel

