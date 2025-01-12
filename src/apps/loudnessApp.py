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
	
	sys.path.append(Path(__file__).parents[1]) # importing modules
	from modules.getLoudness import getLoudness
	from modules.getLoudness import getDf
	
	st.title("loudness") # page title
	
	# file uploader
	inputCols = st.columns(2)
	with inputCols[0]:
		audio = st.file_uploader("upload an audio file", type=["mp3", "wav"], accept_multiple_files=False)
	with inputCols[1]:
		resolution = st.selectbox("resolution setting", ("LOW", "HIGH"), index=1)
	
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
				
				getLoudnessObj =  getLoudness(audioSignal=audioSignal, audioSr=audioSr, resolutionSetting=resolution)
				outs = getLoudnessObj.run()
				loudnessContour, loudnessContourTs, IntegratedLUFS = outs["loudnessContour"], outs["loudnessContourTs"], outs["integratedLUFS"]
				df = getDf(loudnessContour, loudnessContourTs, IntegratedLUFS)
				fig = getLoudnessObj.plot(show=False, title=f"{audioName} at {audioSr} hz")
					
				df["shortTimeLUFS"] = df["shortTimeLUFS"].apply(lambda x: np.round(x, 2)) # round off LUFS values
				csv = df.to_csv(index=False)  # convert the DataFrame to csv (without index)
				# a download button for the CSV
				st.download_button(
					label="↓ CSV",
					data=csv,
					file_name=f"{audioName}-{resolution.lower()}Res.csv",
					mime="text/csv"
				)
				st.pyplot(fig)