import numpy as np
from ibllib.io.extractors.biased_trials import BiasedTrials


class TrialsTableNeuromodulator(BiasedTrials):
    def __init__(self, *args, **kwargs):
        """Include custom trial variables in `var_names` property.

        NB: In order to save to file and register these variables, we would need a custom ibllib.pipes.tasks.Task with
        the correct output file names.
        """
        super().__init__(*args, **kwargs)
        self.save_names = BiasedTrials.save_names + (None, None)
        self.var_names = BiasedTrials.var_names + ('omit_feedback', 'exit_state')

    def _extract(self, *args, **kwargs):
        out = super(TrialsTableNeuromodulator, self)._extract(*args, **kwargs)
        out[0]['omit_feedback'] = np.array([t['omit_feedback'] for t in self.bpod_trials])
        out[0]['exit_state'] = np.array([t['behavior_data']['States timestamps']['exit_state'][0][0] for t in self.bpod_trials])
        return out
