## %% Example of a trials table extraction for Neuromodulator trials
from projects.neuromodulators import TrialsTableNeuromodulator

session_path = "/home/olivier/Downloads/D6/2023-02-08/001"
extractor = TrialsTableNeuromodulator(session_path)
extractor.extract(save=True)
