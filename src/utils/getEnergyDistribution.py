"""
author: ankit anand
created on: 17/12/24
"""

from pathlib import Path
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy

from utils.adapFilter import applyAdaptationFilter # to find the derivative using two gaussians

audioSignal, audioSr = sf.read(file=Path("../tmp/songSamplesSRGM/INH100004500.mp3")) # read audio file

if len(audioSignal.shape) == 2: # stereo
	audioSignal = np.mean(audioSignal, axis=1) # convert to mono
	
audioSignal = audioSignal[40*audioSr:50*audioSr] # one part of the song

N = int(0.05 * audioSr) # 50 ms
H = int(0.01 * audioSr) # 10 ms
Nfft = N
#Nfft = int(2**np.ceil(np.log2(N))) # power of 2

eps = np.finfo(float).eps # 2.22e-16

spec = librosa.stft(y=audioSignal, n_fft=Nfft, win_length=N, window="hann", hop_length=H)

powerSpec = np.abs(spec)**2 # power spec 
powerSpec = powerSpec / np.max(powerSpec) # normalise [0 to 1]
powerSpec = np.clip(powerSpec, a_min=eps, a_max=None) # making sure that the range is [eps, 1]
powerSpecdB = 10*np.log10(powerSpec) # get values in dB [-150, 0]

print(f"spec shape: {powerSpec.shape}")
print(f"eps: {eps}")
print(f"N={N}, H={H}")
print(f"min: {np.min(powerSpec)}, max: {np.max(powerSpec)}, mean: {np.mean(powerSpec)}, median: {np.median(powerSpec)}")
print(f"min: {np.min(powerSpecdB)}, max: {np.max(powerSpecdB)}, mean: {np.mean(powerSpecdB)}, median: {np.median(powerSpecdB)}")

for t in range(powerSpecdB.shape[1]):
	powerSpecdB[:,t] = scipy.signal.medfilt(volume=powerSpecdB[:,t], kernel_size=51) # 21 samples => 420 hz

ts = np.arange(powerSpec.shape[1]) * (H/audioSr)
fs = np.arange(powerSpec.shape[0]) * (audioSr/Nfft)

ders = np.zeros_like(powerSpecdB) # (frequency, timestamp) derivative along frequency dimension for each frame
peaks = np.zeros_like(powerSpecdB) # making it a hot vector for peaks
peaksArray = np.zeros(powerSpecdB.shape[1])
proms = []

for t in range(powerSpecdB.shape[1]): # go through each time frame t
	der, _ = applyAdaptationFilter(signal=powerSpecdB[:, t], signalSr=int(audioSr/Nfft), filterLength=5) # find smoothened derivative
	der = der/np.max(np.abs(der)) # max norm [0, -1]
	der = scipy.signal.medfilt(volume=der, kernel_size=31) # 31 samples => 420 hz
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

#print(f"Cutoff Frequency: {np.mod(peaksArray)}")

fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 10))
img1 = ax[0].imshow(powerSpecdB, aspect="auto", cmap="gray_r", extent=[ts[0], ts[-1], fs[0], fs[-1]], origin="lower")
img2 = ax[1].imshow(ders, aspect="auto", cmap="gray", extent=[ts[0], ts[-1], fs[0], fs[-1]], origin="lower")
#img3 = ax[2].imshow(peaks, aspect="auto", cmap="gray", extent=[ts[0], ts[-1], fs[0], fs[-1]], origin="lower")
#ax[3].plot(ts, peaksArray)
ax[2].hist(peaksArray, bins=400, color='blue', alpha=0.7)

fig.colorbar(img1, ax=ax[0], orientation="vertical")
fig.colorbar(img2, ax=ax[1], orientation="vertical")

ax[0].set_title(f"spectrogram")
ax[1].set_title(f"derivative")

ax[0].set_ylabel("freq (hz)")
ax[0].set_xlabel("time (sec)")

ax[1].set_ylabel("freq (hz)")
ax[1].set_xlabel("time (sec)")

ax[2].set_xlabel("freq (hz)")

#plt.tight_layout()
#plt.show()

#fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(15, 10))
#
#for i in range(3):
#	timeIdx = np.random.randint(0, powerSpec.shape[1])  # random time index
#	
#	# plot the power spectrum at the chosen time index
##	ax[2 * i].scatter(fs, powerSpecdB[:, timeIdx], s=1, color="black", alpha=0.4)
#	ax[2 * i].plot(fs, powerSpecdB[:, timeIdx])
#	ax[2 * i].set_title(f"power spectrum at time index {timeIdx}")
#	ax[2 * i].grid()
#	
#	# plot the derivative at the same time index
#	ax[2 * i + 1].plot(fs, ders[:, timeIdx], color="blue")
#	ax[2 * i + 1].set_title(f"derivative at time index {timeIdx}")
#	ax[2 * i + 1].grid()

plt.tight_layout()
plt.savefig("./INH100004500.png") # save it to output
plt.show()


#E = np.sum(np.abs(cqt)**2, axis=1)
#
#plt.figure(figsize=(15,3))
#plt.plot(fs, E)
#plt.show()
#print(E.shape)
#fs = np.arange(cqt.shape[0]) * (audioSr/)
#plt.figure(figsize)

