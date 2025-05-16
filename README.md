# Audio Features Analysis
A Streamlit-based web application <a href="https://audiofeatures.streamlit.app">link</a> for analyzing various audio features from uploaded audio files.

## Project Overview
This application provides tools to analyze several audio features:

- Loudness Analysis (Completed): Measures short-time loudness using the BS.1770 standard
- Cutoff Frequency Analysis (Completed): Detects high-frequency cutoffs in audio files
- Vocal Chorus Detection (Planned): Structure in place but implementation pending
- Beat Start Detection (Planned): Structure in place but implementation pending
- BPM Analysis (Planned): Structure in place but implementation pending


## 1. Loudness Analysis
The loudness analyzer computes short-time LUFS (Loudness Units Full Scale) of an audio signal using the BS.1770 standard.

**Key Hyperparameters:**
Block Size: 0.4s (Standard block size as per BS.1770)

**Resolution Settings:**
HIGH: 1s window size with 1s hop size (no overlap)
LOW: 5s window size with 5s hop size (no overlap)

**The module outputs:**
Loudness contour over time
Integrated LUFS for the entire track
Visualization of waveform and loudness
CSV export of loudness data

## 2. Cutoff Frequency Analysis
This module detects high-frequency cutoffs in audio files, which can be useful for determining audio quality or identifying low-pass filtering.

**Key Hyperparameters:**
Window Size: 0.5s (N = 0.5 * sampling rate) # longer window as we do not need temporal resolution for the task
Hop Size: 0.5s (no overlap)
FFT Size: Power of 2 for efficient computation

Adaptation Filter:
tau1 = 3 * sampling rate
tau2 = 3 * sampling rate
d1 = 1 * sampling rate
d2 = 1 * sampling rate

Peak Detection: Prominence threshold of 0.2
Frequency Resolution: 100 Hz bins for histogram analysis

**The module outputs:**
Energy distribution across frequency bands
High cutoff frequency estimation
Spectrogram visualization with cutoff overlay

