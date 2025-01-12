"""
author: ankit anand
created on: 09/01/25
"""

import numpy as np
from pathlib import Path
import librosa
from pprint import pprint as pp


#UTILS
import scipy
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
	tau1 = int(3 * signalSr) # standard deviation
	d1 = int(1 * signalSr) # distance from the center
	tau2 = int(3 * signalSr) # standard deviation
	d2 = int(1 * signalSr) # distance from the center
	
	kernel = np.zeros(2 * filterLength)
	t = np.arange(-filterLength, +filterLength+1) 
	kernel = (1/(tau1*np.sqrt(2*np.pi))) * np.exp(-(t-d1)**2/(2*tau1**2)) - (1/(tau2*np.sqrt(2*np.pi))) * np.exp(-(t+d2)**2/(2*tau2**2))
	kernel =  np.exp(-(t-d1)**2/(2*tau1**2)) - np.exp(-(t+d2)**2/(2*tau2**2))
	kernel /= np.sum(np.abs(kernel)) # normalise the kernal
	
	# apply the biphasic filter using convolution
	der = scipy.signal.convolve(signal, kernel[::-1], mode='same') # reversed to perform convolution in the right orientation
	der[der>0] = 0 # for this task we are only interested in drop in intensity
	
	return der, kernel


#--------------------------FUNCTION OBJECT---------------------------
class getCutOffFreq:
	
	def __init__(self, audioSignal:np.ndarray, audioSr:int):
		"""set parameters"""
		self.inps = {
			"audioSignal": audioSignal,
			"audioSr": audioSr
		}
		self.outs = {
			"energyDistribution": None,
			"highCutOff": None
		} # this will be returned on call
		self.debugs = {
			
		} # variables to be used for plotting/debugging
		
		self.debugMode = False
		self.ran = False

		
	def info(self):
		"""returns a dict containing all the higher level info about the function"""
		return {
			"desc": "to find the low and high cut off frequencies of an audio signal",
			"args": [
				{"name": "audioSignal", "type": np.ndarray, "desc": "audio signal for which we want to find cut off frequencies"},
				{"name": "audioSr", "type": int, "desc": "sampling rate of audio"}
			],
			"returns": [
				{"name": "energyDistribution" ,"type": np.ndarray, "desc": "energy distribution across different frequency bands"},
				{"name": "highCutOff" ,"type": float, "desc": "frequency (hz) at which there is expected high cut off"},
			]
		}
	
	def __str__(self):
		"""print details about the function in formatted way"""
		
		info = self.info()
		methods = self.methods()
		
		output = []
		output.append("INFO:")
		output.append(f"\tDescription: {info['desc']}")
		output.append("\tArguments:")
		
		for arg in info['args']:
			output.append(f"\t\t - {arg['name']} ({arg['type'].__name__}): {arg['desc']}")
			
		output.append("\tReturns:")
		
		for ret in info['returns']:
			ret_desc = ret['desc'] if ret['desc'] else "No description provided"
			output.append(f"\t\t - {ret['name']} ({ret['type'].__name__}): {ret_desc}")
			
		output.append("\nMETHODS:")
		
		for method in methods:
			output.append(f"  - {method}")
			
		return "\n".join(output)
	
	
	def methods(self):
		"""returns list of all methods"""
		
		return [
				method for method in self.__dir__()
				if callable(getattr(self, method))  # ensure it's a callable (method)
				and not method.startswith("__")  # excludes special methods (e.g., __init__)
				and method != "methods"  # exclude the methods() function itself
			]
	
	def test(self):
		"""runs custom tests on function"""
		pass
		
	def log(self):
		"""log info about function calls, call it from the run function, wherever needed"""
		import logging
		from pathlib import Path # using path library will ensure compatibilty across os
		logging.basicConfig(
			filename=Path(__file__).parents[0]/"logs.txt",
			level=logging.INFO,  # log all messages at INFO level or higher
			format=(
				"%(asctime)s - File: %(filename)s - Function: %(funcName)s "
				"- Line: %(lineno)d - Level: %(levelname)s - Message: %(message)s"
			),
			datefmt="%d %b %Y, %H:%M:%S",
		)
		return logging.getLogger()
				
	def validate(self):
		"""validate inputs"""
		pass
			
	def run(self, debugMode=False):
		"""main implementation of the code"""
		
		self.ran = True # flag to check if run command was called
		self.debugMode = debugMode # turn on debug mode
		self.validate() # inputs validation
		
		audioSignal = self.inps["audioSignal"]
		audioSr = self.inps["audioSr"]
		
		N = int(0.5 * audioSr) # choosing fairly long window size 
		H = int(0.5 * audioSr) # no overlap
		Nfft = int(2**np.ceil(np.log2(N))) # power of 2 for efficient FFT computation
		
		eps = np.finfo(float).eps # 2.22e-16
		
		spec = librosa.stft(y=audioSignal, n_fft=Nfft, win_length=N, window="hann", hop_length=H)
		
		downsample = 4 # since we are taking longer window, we downsample it by a factor of 4
		spec = spec[::downsample,:] # downgrade
		
		powerSpec = np.abs(spec)**2 # power spec 
		powerSpec = powerSpec / np.max(powerSpec) # normalise [0 to 1]
		powerSpec = np.clip(powerSpec, a_min=eps, a_max=None) # making sure that the range is [eps, 1]
		powerSpecdB = 10*np.log10(powerSpec) # get values in dB [-150, 0]
		
		ts = np.arange(powerSpec.shape[1]) * (H/audioSr)
		fs = np.arange(powerSpec.shape[0]) * (audioSr/(Nfft/downsample))
		
		ders = np.zeros_like(powerSpecdB) # (frequency, timestamp) derivative along frequency dimension for each frame
		peaks = np.zeros_like(powerSpecdB) # making it a hot vector for peaks
		peaksArray = np.zeros(powerSpecdB.shape[1])
		proms = []
		
		freqAxisSr = int(audioSr/(Nfft/downsample))
		
		for t in range(powerSpecdB.shape[1]): # finding the smoothened derivative along frequency dimension, we go through each time frame t
			der, _ = applyAdaptationFilter(signal=powerSpecdB[:,t], signalSr=freqAxisSr, filterLength=int(10*freqAxisSr)) # find smoothened derivative along freq axis
			ders[:, t] = der
		
		for t in range(powerSpecdB.shape[1]): # go through each time frame t
			# peak peaking
			peak, props = scipy.signal.find_peaks(-ders[:,t], prominence=0.2)
			prom = props["prominences"]
			if len(peak) != 0:
				maxPromsIdx = np.argmax(prom) # find the freq index where there is maximum drop
				
				maxPeakIdx = peak[maxPromsIdx]
				peaksArray[t] = fs[maxPeakIdx]
				peaks[maxPeakIdx,t] = 1
				
		binEdges = np.arange(0, audioSr//2, 100) # resolution is kept at 1000, note that this is fine for looking at higher frequencies
		energyDistribution, _ = np.histogram(peaksArray, bins=binEdges) # energy distribution at 100 hz frequency resolution
		highCutOff = binEdges[np.argmax(energyDistribution)] # high cutoff value
		
		self.outs["energyDistribution"] = energyDistribution
		self.outs["highCutOff"] = highCutOff
		
		if self.debugMode:
			self.debugs["powerSpecdB"] = powerSpecdB
			self.debugs["ders"] = ders
			self.debugs["peaksArray"] = peaksArray
			self.debugs["fs"] = fs
			self.debugs["ts"] = ts

		return self.outs
	
	def plot(self, show=False):
		"""plot to debug"""
		
		if not self.ran:
			raise PermissionError("run method needs to be called before plotting")
		if not self.debugMode:
			raise PermissionError("run method needs to be called with debugMode True before plotting")
		
		import matplotlib.pyplot as plt
		
		powerSpecdB = self.debugs["powerSpecdB"]
		ders = self.debugs["ders"]
		peaksArray = self.debugs["peaksArray"]
		fs = self.debugs["fs"]
		ts = self.debugs["ts"]
		
		fig, ax = plt.subplots(figsize=(14, 3))
		img1 = ax.imshow(powerSpecdB, aspect="auto", cmap="gray_r", extent=[ts[0], ts[-1], fs[0], fs[-1]], origin="lower")
		ax.plot(ts, peaksArray, color='blue', linewidth=1.5)
		fig.colorbar(img1, ax=ax, orientation="vertical")
		ax.set_title(f"spectrogram")
		ax.set_ylabel("freq (hz)")
		ax.set_xlabel("time (sec)")
		
		if show:
			plt.tight_layout()
			plt.show()
		
		return fig
		
	def saveOutput(self, outputDp:Path=None, dirname:str=None):
		"""save debug/output files"""
		
		if not self.ran:
			raise PermissionError("run method needs to be called before saving output")
			
		pass