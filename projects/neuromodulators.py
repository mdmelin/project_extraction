import numpy as np
from ibllib.io.extractors.biased_trials import BiasedTrials
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes


class TrialsTableNeuromodulator(BaseBpodTrialsExtractor):
    """Extract neuromodulator task events.

    Include a couple custom trial variables in `var_names` property.

    NB: In order to save to file and register these variables, we would need a custom ibllib.pipes.tasks.Task with
    the correct output file names.
    """
    save_names = BiasedTrials.save_names + (None, None)
    var_names = BiasedTrials.var_names + ('omit_feedback', 'exit_state')

    def _extract(self, *args, **kwargs):
        # Extract common biased choice world datasets
        out, _ = run_extractor_classes(
            [BiasedTrials], session_path=self.session_path, bpod_trials=self.bpod_trials,
            settings=self.settings, save=False, task_collection=self.task_collection)
        out[0]['omit_feedback'] = np.array([t['omit_feedback'] for t in self.bpod_trials])
        out[0]['exit_state'] = np.array([t['behavior_data']['States timestamps']['exit_state'][0][0] for t in self.bpod_trials])
        return {k: out[k] for k in self.var_names}  # Ensures all datasets present and ordered
