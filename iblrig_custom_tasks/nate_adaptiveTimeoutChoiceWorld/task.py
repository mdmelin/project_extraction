"""
This task inherits TrainingChoiceWorldSession with the addition of configurable, adaptive timeouts for incorrect
choices depending on the stimulus contrast.
"""

import logging
from pathlib import Path

import yaml

from iblrig.base_choice_world import TrainingChoiceWorldSession
from iblrig.misc import get_task_arguments
from pybpodapi.state_machine import StateMachine

log = logging.getLogger('iblrig.task')


# read defaults from task_parameters.yaml
with open(Path(__file__).parent.joinpath('task_parameters.yaml')) as f:
    DEFAULTS = yaml.safe_load(f)


class AdaptiveTimeoutStateMachine(StateMachine):

    def __init__(
        self,
        bpod,
        adaptive_delay_nogo,
        adaptive_delay_error
    ):
        super().__init__(bpod)
        self.adaptive_delay_nogo = adaptive_delay_nogo
        self.adaptive_delay_error = adaptive_delay_error


    def add_state(self, **kwargs):
        match kwargs['state_name']:
            case 'nogo':
                pass
            case 'error':
                pass
        super().add_state(**kwargs)


class Session(TrainingChoiceWorldSession):
    protocol_name = 'nate_adaptiveTimeoutChoiceWorld'

    def __init__(
        self,
        *args,
        adaptive_delay_nogo=DEFAULTS['ADAPTIVE_FEEDBACK_NOGO_DELAY_SECS'],
        adaptive_delay_error=DEFAULTS['ADAPTIVE_FEEDBACK_ERROR_DELAY_SECS'],
        **kwargs,
    ):
        self.adaptive_delay_nogo = adaptive_delay_nogo
        self.adaptive_delay_error = adaptive_delay_error
        super().__init__(*args, **kwargs)
        assert len(self.adaptive_delay_nogo) == len(self.task_params.CONTRAST_SET)
        assert len(self.adaptive_delay_error) == len(self.task_params.CONTRAST_SET)

    def _instantiate_state_machine(self, trial_number=None):
        return AdaptiveTimeoutStateMachine(self.bpod, self.adaptive_delay_nogo, self.adaptive_delay_error)

    @staticmethod
    def extra_parser():
        parser = super(Session, Session).extra_parser()
        parser.add_argument(
            '--adaptive_delay_nogo',
            option_strings=['--adaptive_delay_nogo'],
            dest='adaptive_delay_nogo',
            default=DEFAULTS['ADAPTIVE_FEEDBACK_NOGO_DELAY_SECS'],
            nargs='+',
            type=float,
            help='list of delays for no-go condition (contrasts: 1.0, 0.25, 0.125, 0.0625, 0.0)',
        )
        parser.add_argument(
            '--adaptive_delay_error',
            option_strings=['--adaptive_delay_error'],
            dest='adaptive_delay_nogo',
            default=DEFAULTS['ADAPTIVE_FEEDBACK_ERROR_DELAY_SECS'],
            nargs='+',
            type=float,
            help='list of delays for error condition (contrasts: 1.0, 0.25, 0.125, 0.0625, 0.0)',
        )
        return parser


if __name__ == '__main__':  # pragma: no cover
    kwargs = get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()
