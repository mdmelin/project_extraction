"""
This task is a replica of BiasedChoiceWorldSession with the addition of optogenetic stimulation
An `opto_stimulation` column is added to the trials_table, which is a boolean array of length NTRIALS_INIT
The PROBABILITY_OPTO_STIMULATION parameter is used to determine the probability of optogenetic stimulation for each trial

Additionally the state machine is modified to add output TTLs for optogenetic stimulation
"""

import numpy as np
from pybpodapi.protocol import StateMachine

from iblrig.base_choice_world import BiasedChoiceWorldSession
from iblutil.util import setup_logger

log = setup_logger(__name__)

INTERACTIVE_DELAY = 1.0
NTRIALS_INIT = 2000


class OptoStateMachine(StateMachine):
    """
    This class just adds output TTL on BNC2 for defined states
    """
    def __init__(self, bpod, is_opto_stimulation=False, states_opto_ttls=None):
        super().__init__(bpod)
        self.is_opto_trial = is_opto_stimulation
        self.states_opto_ttls = states_opto_ttls or []

    def add_state(self, **kwargs):
        if self.is_opto_stimulation and kwargs['state_name'] in self.states_opto_ttls:
            kwargs.output_actions.append(("BNC2", 255))
        super().add_state(**kwargs)


class Session(BiasedChoiceWorldSession):
    protocol_name = "nate_optoBiasedChoiceWorld"
    extractor_tasks = ['TrialRegisterRaw', 'ChoiceWorldTrials', 'TrainingStatus']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # generates the opto stimulation for each trial
        p = self.task_params['PROBABILITY_OPTO_STIM']
        self.trials_table['opto_stimulation'] = np.random.choice([0, 1], p=[1 - p, p], size=NTRIALS_INIT).astype(bool)

    def _instantiate_state_machine(self, trial_number=None):
        """
        We override this using the custom class OptoStateMachine that appends TTLs for optogenetic stimulation where needed
        :param trial_number:
        :return:
        """
        is_opto_stimulation = self.trials_table.at[trial_number, 'opto_stimulation']
        states_opto_ttls = self.task_params['OPTO_STIM_STATES']
        return OptoStateMachine(self.bpod, is_opto_stimulation=is_opto_stimulation, states_opto_ttls=states_opto_ttls)
