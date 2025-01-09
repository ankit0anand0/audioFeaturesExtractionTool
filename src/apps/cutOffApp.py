"""
author: ankit anand
created on: 03/01/25
"""

def cutOffApp():
	"""
	desc
	----
		app that compute short-time loudness
	args
	----
		None
	returns
	-------
		None

	"""
	from pathlib import Path
	import sys
	import streamlit as st
	import matplotlib.pyplot as plt
	import numpy as np
	import io
	import librosa
	
	sys.path.append(Path(__file__).parents[1])
	from modules.getCutOffFreq import getCutOffFreq
	
	st.title("cut off") # page title
	
	# file uploader
	inputCols = st.columns(2)
	with inputCols[0]:
		audio = st.file_uploader("upload an audio file", type=["mp3", "wav"], accept_multiple_files=False)
		
	if audio is not None:
		audioData = audio.getvalue()
		audioName = audio.name.split(".")[0]
		isAudioLoaded = False
		with st.spinner("loading audio"):
			try:
				audioBuffer = io.BytesIO(audioData)  # convert binary data to a file-like object
				audioSignal, audioSr = librosa.load(audioBuffer, sr=None)  # read the audio as numpy array
				st.audio(audioData, format="audio/")
				isAudioLoaded = True
			except Exception as e:
				isAudioLoaded = False
				st.error("audio loading issue")
				
		if isAudioLoaded:
			with st.spinner("computing high cutoff"):
				getCutOffFreqObj =  getCutOffFreq(audioSignal=audioSignal, audioSr=audioSr)
				outs = getCutOffFreqObj.run(debugMode=True)
				energyDistribution, highCutOff = outs["energyDistribution"], outs["highCutOff"]
				fig = getCutOffFreqObj.plot(show=False)
				st.text(f"high cutoff: {highCutOff} hz")
				st.pyplot(fig)