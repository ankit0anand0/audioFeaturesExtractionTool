"""
author: ankit anand
created on: 09/01/25
"""

import numpy as np
from pathlib import Path
import pyloudnorm as pyln

#--------------------------SETTINGS---------------------------
HIGH_RESOLUTION_SETTINGS = {
	"name": "highres",
	"blockSize": 0.4, # standard block size as per BS.1770
	"winSizeSec": 1., # 1 s window-size
	"hopSizeSec": 1. # no overlap
}

LOW_RESOLUTION_SETTINGS = {
	"name": "lowres",
	"blockSize": 0.4, # standard block size as per BS.1770
	"winSizeSec": 5., # 5 sec window 
	"hopSizeSec": 5. # no overlap
}

#--------------------------UTILS---------------------------
def secondsTommss(x:float=None, _=None):
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

def getDf(loudnessContour, loudnessContourTs, integratedLUFS):
	"""
	desc
	----
		create a dataframe to share perceived loudness contour with user
	
	args
	----
		loudnessContour: 
		loudnessContoutTs:
		integratedLUFS:
	returns
	-------
		df:
	"""
	import pandas as pd
	df = pd.DataFrame({
		"timestamp": np.round(loudnessContourTs, 2),
		"shortTimeLUFS": loudnessContour
	})
	
	return df

#--------------------------FUNCTION OBJECT---------------------------
class getLoudness:
	
	def __init__(self, audioSignal:np.ndarray=None, audioSr:int=None, resolutionSetting:dict=None):
		"""set parameters"""
		"""set parameters"""
		self.inps = {
			"audioSignal": audioSignal,
			"audioSr": audioSr,
			"resolutionSetting": resolutionSetting
		}
		self.outs = {
			"loudnessContour": None,
			"loudnessContourTs": None,
			"integratedLoudness": None
		} # this will be returned on call
		self.debugs = {
			
		} # variables to be used for plotting/debugging
		
	def info(self):
		"""returns a dict containing all the higher level info about the function"""
		return {
			"desc": "to compute short-time LUFS of an audio signal",
			"args": [
				{"name": "audioSignal", "type": np.ndarray, "desc": "audio signal to compute short-time LUFS"},
				{"name": "audioSr", "type": int, "desc": "sampling rate of the audio signal"},
				{"name": "resolutionSetting", "type": dict, "desc": "resolution setting for the parameters"},
			],
			"returns": [
				{"name": "loudnessContour","type": np.ndarray, "desc": "loudness contour array"},
				{"name": "loudnessContourTs","type": np.ndarray, "desc": "loudness contour timestamp"},
				{"name": "integratedLoudness","type": float, "desc": ""}
			]
		}
	
	def methods(self):
		"""returns list of all methods"""
		return [method for method in self.__dir__() if not method.startswith("__") and method != "methods"]
	
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
		assert self.inps["audioSignal"].shape != 0, "audio length can't be 0"
		
			
	def run(self):
		"""main implementation of the code"""
		
		self.validate() # validate all the inputs
		audioSignal = self.inps["audioSignal"]
		audioSr = self.inps["audioSr"]
		resolutionSetting = self.inps["resolutionSetting"]
		
		if len(audioSignal.shape) == 2:
			audioSignal = np.mean(audioSignal, axis=1) # stereo to mono
			
		loudnessMeter = pyln.Meter(audioSr, block_size=resolutionSetting["blockSize"]) # BS.1770 loudness meter object (400 ms block size)
		integratedLUFS = loudnessMeter.integrated_loudness(audioSignal) # this is computed to check against spotify API
		
		winSizeSamples = int(resolutionSetting["winSizeSec"] * audioSr) # window size in samples
		hopSizeSamples = int(resolutionSetting["hopSizeSec"] * audioSr) # hop size in samples
		
		loudnessContour = np.array([]) # to store short time perceived loudness
		
		startIdx = 0
		while startIdx < audioSignal.shape[0]:
			segment = audioSignal[startIdx:startIdx+winSizeSamples] # audio segment for short time analysis
			
			if len(segment) < winSizeSamples: # for the last segment, we pad the segment such that the sgement size remains same
				segment = np.pad(segment, (0, winSizeSamples-len(segment)), mode="constant", constant_values=0) # pad 0s at the end of the sequence
			
			loudness = loudnessMeter.integrated_loudness(segment) # loudness for the segment
			loudnessContour = np.append(loudnessContour, loudness) # append the values
			startIdx += hopSizeSamples # jump by hop
			
		loudnessContour = np.clip(loudnessContour, a_min=-70, a_max=None) # clip silence regions to -70
		loudnessContourTs = np.arange(loudnessContour.shape[0]) * (hopSizeSamples/audioSr) # loudness timestamps (sec)
		
		self.outs["loudnessContour"] = loudnessContour
		self.outs["loudnessContourTs"] = loudnessContourTs
		self.outs["integratedLUFS"] = integratedLUFS
		
		return self.outs
	
	def plot(self, show=False):
		"""plot to debug"""
		
		import matplotlib.pyplot as plt
		
		audioSignal = self.inps["audioSignal"]
		audioSr = self.inps["audioSr"]
		print(audioSignal, audioSr)
		
		loudnessContourTs = self.outs["loudnessContourTs"]
		loudnessContour = self.outs["loudnessContour"]
		integratedLUFS = self.outs["integratedLUFS"]
		
		print(loudnessContourTs)
		
		audioDur = audioSignal.shape[0]/audioSr # for xlim
		audioTs = np.arange(audioSignal.shape[0])/audioSr # timestamp
		
		fig, ax = plt.subplots(figsize=(15, 6), nrows=2, ncols=1, sharex=True)
		fig.suptitle(f"untitled")
		
		# waveform
		ax[0].plot(audioTs, audioSignal)
		ax[0].set_title("waveform")
		ax[0].set_ylabel("amplitude")
		ax[0].set_xticks(np.arange(0, audioDur, 5))
		ax[0].set_xlim(0, audioDur)
		ax[0].xaxis.set_major_formatter(plt.FuncFormatter(secondsTommss))
		ax[0].tick_params(axis='x', labelrotation=90)
		ax[0].grid()
		
		# loudness contour
		ax[1].plot(loudnessContourTs, loudnessContour, color="orange", linewidth=2, marker="o", markersize=3)
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
		
		return fig
	
	def saveLocal(self, outputDp:Path=None, dirname:str=None):
		"""
		desc
		----
			save debug/output files locally
		args
		----
			outputDp:
			dirname:
		returns
		-------
			None
		"""
		import matplotlib.pyplot as plt
		loudnessContour, loudnessContourTs, integratedLUFS, = self.run()
		df = getDf(loudnessContour, loudnessContourTs, integratedLUFS)
		fig = self.plot(show=False)

		# save files
		if outputDp is not None and dirname is not None: # we can use this flag to tell if we want to save the files (eg. in webapp, we might not want to save)
			resolution = self.resolutionSetting["name"] # file name should contain this
			(outputDp/dirname).mkdir(exist_ok="True") # create a folder to store all computed files
			df.to_csv(outputDp/dirname/f"{resolution}.csv", index=False) # save csv
			plt.savefig(outputDp/dirname/f"{resolution}.png") # save plot