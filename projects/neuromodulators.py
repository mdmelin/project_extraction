import numpy as np
from ibllib.io.extractors.biased_trials import BiasedTrials
from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsBpod


class ChoiceWorldNeuromodulators(ChoiceWorldTrialsBpod):
    def _run(self, update=True):
        """
        Extracts an iblrig training session
        """
        save_path = self.session_path.joinpath(self.output_collection)
        extractor = TrialsTableNeuromodulator(session_path=self.session_path, task_collection=self.task_collection, save_path=save_path)
        out, fil = extractor.extract()
        return fil


class TrialsTableNeuromodulator(BiasedTrials):

    def _extract(self, *args, **kwargs):
        out = super(TrialsTableNeuromodulator, self)._extract(*args, **kwargs)
        out[0]['omit_feedback'] = np.array([t['omit_feedback'] for t in self.bpod_trials])
        out[0]['exit_state'] = np.array([t['behavior_data']['States timestamps']['exit_state'][0][0] for t in self.bpod_trials])
        return out
