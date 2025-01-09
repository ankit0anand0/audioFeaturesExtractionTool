"""
author: ankit anand
created on: 17/12/24
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy

from adapFilter import applyAdaptationFilter # to find the derivative using two gaussians

def getEnergyDistribution(audioSignal:np.ndarray=None, audioSr:int=None):
	"""
	desc
	----
		to find the low and high cut off frequencies of an audio signal
	args
	----
		audioSignal: np.ndarray
			audio signal for which we want to find cut off frequencies
		audioSr: int
			sampling rate of audio
	returns
	-------
		fig: plt.figure
			plot to observe frame level cut offs
		lowCutOff: int
			low cut off value
		highCutOff: int
			high cut off value
	"""	

	N = int(0.5 * audioSr) # long window size
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
	
	print(f"spec shape: {powerSpec.shape}")
	print(f"eps: {eps}")
	print(f"N={N}, H={H}, Nfft={Nfft}")
	print(f"min: {np.min(powerSpec)}, max: {np.max(powerSpec)}, mean: {np.mean(powerSpec)}, median: {np.median(powerSpec)}")
	print(f"min: {np.min(powerSpecdB)}, max: {np.max(powerSpecdB)}, mean: {np.mean(powerSpecdB)}, median: {np.median(powerSpecdB)}")
	
#	smoothenedKernelSize = int(420*(Nfft/audioSr)) # smoothing 420 hz
#	smoothenedKernelSize = smoothenedKernelSize+1 if smoothenedKernelSize//2 == 0 else smoothenedKernelSize # make sure it is odd to perform median filtering
#	for t in range(powerSpecdB.shape[1]): # smoothing along frequency dimension
#		powerSpecdB[:,t] = scipy.signal.medfilt(volume=powerSpecdB[:,t], kernel_size=smoothenedKernelSize) # 420 hz
	
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
	hist, _ = np.histogram(peaksArray, bins=binEdges)
	maxBinIndex = np.argmax(hist)
	binCenter = binEdges[maxBinIndex]
	print(binCenter)
	
	fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 8))
	img1 = ax[0].imshow(powerSpecdB, aspect="auto", cmap="gray_r", extent=[ts[0], ts[-1], fs[0], fs[-1]], origin="lower")
	img2 = ax[1].imshow(ders, aspect="auto", cmap="gray", extent=[ts[0], ts[-1], fs[0], fs[-1]], origin="lower")
	ax[0].plot(ts, peaksArray, color='blue', linewidth=2)
	
	fig.colorbar(img1, ax=ax[0], orientation="vertical")
	fig.colorbar(img2, ax=ax[1], orientation="vertical")
	
	ax[0].set_title(f"spectrogram")
	ax[1].set_title(f"derivative")
	
	ax[0].set_ylabel("freq (hz)")
	ax[0].set_xlabel("time (sec)")
	
	ax[1].set_ylabel("freq (hz)")
	ax[1].set_xlabel("time (sec)")
	
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	from pathlib import Path
	import soundfile as sf
	audioSignal, audioSr = sf.read(file=Path("../../tmp/espresso.mp3")) # read audio file
	if len(audioSignal.shape) == 2: # stereo
		audioSignal = np.mean(audioSignal, axis=1) # convert to mono
	
#	audioSignal = audioSignal[0:int(10*audioSr)]
	getEnergyDistribution(audioSignal=audioSignal, audioSr=audioSr)
	