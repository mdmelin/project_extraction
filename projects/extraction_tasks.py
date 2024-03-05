"""
Custom behaviour extractor tasks for personal projects.

The task class name(s) should be added to the

NB: This may need changing in the future if one of these modules requires optional dependencies.
"""
from projects.neuromodulators import ChoiceWorldNeuromodulators
from projects.samuel_cuedBiasedChoiceWorld import CuedBiasedTrials, CuedBiasedTrialsTimeline
from projects.nate_optoBiasedChoiceWorld import OptoTrialsNidq, OptoTrialsBpod
