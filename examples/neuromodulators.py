## %% Example of a trials table extraction for Neuromodulator trials
from projects.neuromodulators import TrialsTableNeuromodulator


session_path = "/datadisk/gdrive/2023/02_Neuromodulators/D6/2023-02-08/001"
extractor = TrialsTableNeuromodulator(session_path)
_, data_files = extractor.extract(save=True)


## %% Loads the data afterwards
import pandas as pd
trials = pd.read_parquet(data_files[0])

assert set(['exit_state', 'omit_feedback']).issubset(set(trials.keys()))
