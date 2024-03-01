"""
This task is a replica of BiasedChoiceWorldSession with the addition of optogenetic stimulation
An `opto_stimulation` column is added to the trials_table, which is a boolean array of length NTRIALS_INIT
The PROBABILITY_OPTO_STIMULATION parameter is used to determine the probability of optogenetic stimulation for each trial

Additionally the state machine is modified to add output TTLs for optogenetic stimulation
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Literal

from pybpodapi.protocol import StateMachine

from iblrig.base_choice_world import BiasedChoiceWorldSession
from iblutil.util import setup_logger
import iblrig

log = setup_logger(__name__)

INTERACTIVE_DELAY = 1.0
NTRIALS_INIT = 2000

# read defaults from task_parameters.yaml
with open(Path(__file__).parent.joinpath('task_parameters.yaml')) as f:
    DEFAULTS = yaml.safe_load(f)


class OptoStateMachine(StateMachine):
    """
    This class just adds output TTL on BNC2 for defined states
    """

    def __init__(self, bpod, is_opto_stimulation=False, states_opto_ttls=None):
        super().__init__(bpod)
        self.is_opto_stimulation = is_opto_stimulation
        self.states_opto_ttls = states_opto_ttls or []

    def add_state(self, **kwargs):
        if self.is_opto_stimulation and kwargs['state_name'] in self.states_opto_ttls:
            kwargs['output_actions'].append(('BNC2', 255))
        super().add_state(**kwargs)


class Session(BiasedChoiceWorldSession):
    protocol_name = 'nate_optoBiasedChoiceWorld'

    def __init__(
        self,
        *args,
        probability_opto_stim: float = DEFAULTS['PROBABILITY_OPTO_STIM'],
        contrast_set_probability_type: Literal['skew_zero', 'uniform'] = DEFAULTS['CONTRAST_SET_PROBABILITY_TYPE'],
        opto_stim_states: list[str] = DEFAULTS['OPTO_STIM_STATES'],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        print(probability_opto_stim)
        print(contrast_set_probability_type)
        print(opto_stim_states)
        self.task_params['CONTRAST_SET_PROBABILITY_TYPE'] = contrast_set_probability_type
        self.task_params['OPTO_STIM_STATES'] = opto_stim_states
        self.task_params['PROBABILITY_OPTO_STIM'] = probability_opto_stim

        # generates the opto stimulation for each trial
        self.trials_table['opto_stimulation'] = np.random.choice(
            [0, 1], p=[1 - probability_opto_stim, probability_opto_stim], size=NTRIALS_INIT
        ).astype(bool)

    def _instantiate_state_machine(self, trial_number=None):
        """
        We override this using the custom class OptoStateMachine that appends TTLs for optogenetic stimulation where needed
        :param trial_number:
        :return:
        """
        is_opto_stimulation = self.trials_table.at[trial_number, 'opto_stimulation']
        states_opto_ttls = self.task_params['OPTO_STIM_STATES']
        return OptoStateMachine(self.bpod, is_opto_stimulation=is_opto_stimulation, states_opto_ttls=states_opto_ttls)

    @staticmethod
    def extra_parser():
        """:return: argparse.parser()"""
        parser = super(Session, Session).extra_parser()
        parser.add_argument(
            '--probability_opto_stim',
            option_strings=['--probability_opto_stim'],
            dest='probability_opto_stim',
            default=DEFAULTS['PROBABILITY_OPTO_STIM'],
            type=float,
            help=f'probability of opto-genetic stimulation (default: {DEFAULTS["PROBABILITY_OPTO_STIM"]})',
        )
        parser.add_argument(
            '--contrast_set_probability_type',
            option_strings=['--contrast_set_probability_type'],
            dest='contrast_set_probability_type',
            default=DEFAULTS['CONTRAST_SET_PROBABILITY_TYPE'],
            type=str,
            choices=['skew_zero', 'uniform'],
            help=f'probability type for contrast set (default: {DEFAULTS["CONTRAST_SET_PROBABILITY_TYPE"]})',
        )
        parser.add_argument(
            '--opto_stim_states',
            option_strings=['--opto_stim_states'],
            dest='opto_stim_states',
            default=DEFAULTS['OPTO_STIM_STATES'],
            nargs='+',
            type=str,
            help=f'list of the state machine states where opto stim should be delivered',
        )
        return parser


if __name__ == '__main__':  # pragma: no cover
    kwargs = iblrig.misc.get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()
