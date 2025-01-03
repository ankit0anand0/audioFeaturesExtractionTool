"""
author: ankit anand
created on: 23/12/24
"""

def loudnessApp():
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
	from utils.getPerceivedLoudness import getPerceivedLoudness
	
	#---------------------------SETTINGS------------------------------
	HIGH_RESOLUTION_SETTINGS = {
		"name": "highres",
		"blockSize": 0.4, # standard block size as per BS.1770
		"winSizeSec": 1., # 600 ms window-size
		"hopSizeSec": 1. # No overlap
	}
	
	LOW_RESOLUTION_SETTINGS = {
		"name": "lowres",
		"blockSize": 0.4, # standard block size as per BS.1770
		"winSizeSec": 5., # 1 sec window 
		"hopSizeSec": 5. # No overlap
	}
	
	
	st.title("loudness") # page title
	
	# file uploader
	inputCols = st.columns(2)
	with inputCols[0]:
		audio = st.file_uploader("upload an audio file", type=["mp3", "wav"], accept_multiple_files=False)
	with inputCols[1]:
		resolution = st.selectbox("resolution (sec)", (1, 5))
	
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
			with st.spinner("computing loudness"):
				if resolution == 1: # high resolution
					df, fig = getPerceivedLoudness(audioSignal, audioSr, None, HIGH_RESOLUTION_SETTINGS, audioName)
				elif resolution == 5: # low resolution
					df, fig = getPerceivedLoudness(audioSignal, audioSr, None, LOW_RESOLUTION_SETTINGS, audioName)
				
				df["shortTimeLUFS"] = df["shortTimeLUFS"].apply(lambda x: np.round(x, 2)) # round off LUFS values
				csv = df.to_csv(index=False)  # convert the DataFrame to csv (without index)
				# a download button for the CSV
				st.download_button(
					label="download csv",
					data=csv,
					file_name=f"{audioName}-{resolution}secLoudness.csv",
					mime="text/csv"
				)
				st.pyplot(fig)