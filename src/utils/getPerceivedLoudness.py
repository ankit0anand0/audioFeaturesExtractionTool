"""
author: ankit anand
created on: 17/12/24
"""

from pathlib import Path
import pyloudnorm as pyln
import soundfile as sf
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import pandas as pd


#---------------------------SETTINGS------------------------------
HIGH_RESOLUTION_SETTINGS = {
	"name": "highres",
	"blockSize": 0.4, # standard block size as per BS.1770
	"winSizeSec": 1., # 600 ms window-size
	"hopSizeSec": 1. # no overlap
}

LOW_RESOLUTION_SETTINGS = {
	"name": "lowres",
	"blockSize": 0.4, # standard block size as per BS.1770
	"winSizeSec": 5., # 1 sec window 
	"hopSizeSec": 5. # no overlap
}


#---------------------------FUNCTIONS------------------------------
def seconds_to_mmss(x:float=None, _=None):
	"""
	desc
	----
		format x-axis labels from sec to mm:ss
	args
	----
		x: int
			time in sec
	returns
	-------
		_: str
			time in mm:ss format
	"""
	mins = int(x // 60)
	secs = int(x % 60)
	return f"{mins:02}:{secs:02}"

def getPerceivedLoudness(audioSignal:np.ndarray=None, audioSr:int=None, outputDp:Path=None, resolutionSetting:dict=None, audioFn:str="Untitled"):
	"""
	desc
	----
		compute the short-time perceived loudness of an audio signal and also get the plot
	args
	----
		audioSignal: np.ndarray
			audio signal
		audioSr: int
			sampling rate of audio
		outputDp: Path
			dirpath to dump results (csv and png)
		resolutionSetting: dict
			resolution parameters
	returns
	-------
		df: pd.DataFrame
			dataframe containing all the loudness values at regular intervals based on resolutionSetting
		fig: plt.figure
			plot of audio waveform and loudness contour
	"""

	audioDur = audioSignal.shape[0]/audioSr # use to put xlim on plots
	audioTs = np.arange(audioSignal.shape[0])/audioSr # use as xs for plots
	
	if len(audioSignal.shape) == 2:
		audioSignal = np.mean(audioSignal, axis=1) # if stereo, convert to mono
	
	loudnessMeter = pyln.Meter(audioSr, block_size=resolutionSetting["blockSize"]) # create BS.1770 loudness meter (400 ms block size)
	integratedLUFS = loudnessMeter.integrated_loudness(audioSignal) # this is computed to check against spotify API
	
	winSizeSamples = int(resolutionSetting["winSizeSec"] * audioSr) # window size in samples
	hopSizeSamples = int(resolutionSetting["hopSizeSec"] * audioSr) # hop size in samples
	
	perceivedLoudnessContour = np.array([]) # to store short time perceived loudness
	
	startIdx = 0
	while startIdx < audioSignal.shape[0]:
		segment = audioSignal[startIdx:startIdx+winSizeSamples] # audio segment for short time analysis
		
		if len(segment) < winSizeSamples: # for the last segment, we pad the segment such that the sgement size remains same
			segment = np.pad(segment, (0, winSizeSamples-len(segment)), mode="constant", constant_values=0) # pad 0s at the end of the sequence
			
		loudness = loudnessMeter.integrated_loudness(segment) # to compute loudness for the segment
		perceivedLoudnessContour = np.append(perceivedLoudnessContour, loudness) # keep appending the values
		
		startIdx += hopSizeSamples # move start index by hop size
	
	perceivedLoudnessContour = np.clip(perceivedLoudnessContour, a_min=-70, a_max=None) # clipping silence regions to -70
	perceivedLoudnessContourTs = np.arange(perceivedLoudnessContour.shape[0]) * (hopSizeSamples/audioSr) # loudness timestamps (sec)
	
	# create a dataframe to share perceived loudness contour with user
	df = pd.DataFrame({
		"timestamp": np.round(perceivedLoudnessContourTs, 2),
		"shortTimeLUFS": perceivedLoudnessContour
	})
	
	# plot
	fig, ax = plt.subplots(figsize=(15, 6), nrows=2, ncols=1, sharex=True)
	fig.suptitle(f"{audioFn} at {audioSr} Hz")
	
	# waveform
	ax[0].plot(audioTs, audioSignal)
	ax[0].set_title("waveform")
	ax[0].set_ylabel("amplitude")
	ax[0].set_xticks(np.arange(0, audioDur, 5))
	ax[0].set_xlim(0, audioDur)
	ax[0].xaxis.set_major_formatter(plt.FuncFormatter(seconds_to_mmss))
	ax[0].tick_params(axis='x', labelrotation=90)
	ax[0].grid()
	
	# perceived loudness contour
	ax[1].plot(perceivedLoudnessContourTs, perceivedLoudnessContour, color="orange", linewidth=2, marker="o", markersize=3)
	ax[1].set_title(f"short-time LUFS (integrated: {integratedLUFS:.2f} LUFS)")
	ax[1].set_xlabel("time (mm:ss)")
	ax[1].set_ylabel("loudness (LUFS)")
	ax[1].set_xticks(np.arange(0, audioDur, 5))
	ax[1].set_yticks(np.arange(0, -70, -10))
	ax[1].set_xlim(0, audioDur)
	ax[1].set_ylim(-70, 0)  # ensure limits align with data
	ax[1].tick_params(axis='x', labelrotation=90)
	ax[1].grid()
	
	fig.tight_layout(rect=[0, 0, 1, 0.96])  # space for suptitle
	
	
	# save files
	if outputDp is not None: # we can use this flag to tell if we want to save the files (eg. in webapp, we might not want to save)
		resolution = resolutionSetting["name"] # file name should contain this
		(outputDp/audioFn).mkdir(exist_ok="True") # create a folder to store all computed files
		df.to_csv(outputDp/audioFn/f"{resolution}.csv", index=False) # save csv
		plt.savefig(outputDp/audioFn/f"{resolution}.png") # save plot
	else:
		return df, fig