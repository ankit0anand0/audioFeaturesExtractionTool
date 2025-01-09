"""
author: ankit anand
created on: 23/12/24
"""

import streamlit as st
from apps.loudnessApp import loudnessApp
from apps.cutOffApp import cutOffApp


#setup the page
st.set_page_config(
	page_title="Audio features",
	layout="wide",               # Layout style: "centered" or "wide"
	initial_sidebar_state="auto",    # Sidebar state: "auto", "expanded", or "collapsed"
)

st.sidebar.title("audio features")
app = st.sidebar.selectbox("audio features", ["loudness", "cut-off", "vocal chorus", "beat start", "bpm"], label_visibility="hidden")

# Page selection
if app == "loudness":
	loudnessApp()
elif app == "cut-off":
	cutOffApp()
elif app == "vocal chorus":
	pass
elif app == "beat start":
	pass
elif app == "bpm":
	pass