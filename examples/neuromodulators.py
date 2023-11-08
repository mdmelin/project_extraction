"""
Example of a trials table extraction for Neuromodulator trials
"""
import pandas as pd
from projects.neuromodulators import ChoiceWorldNeuromodulators

session_path = "/datadisk/gdrive/2023/02_Neuromodulators/D6/2023-02-08/001"
session_path = "/datadisk/gdrive/2023/02_Neuromodulators/ZFM-04022/2023-03-24/001"


task = ChoiceWorldNeuromodulators(session_path)






## %% Loads the data afterwards
trials = pd.read_parquet(data_files[0])

assert set(['exit_state', 'omit_feedback']).issubset(set(trials.keys()))
