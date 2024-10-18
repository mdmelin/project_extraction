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
    def add_state(self, **kwargs):
        super().add_state(**kwargs)


class Session(TrainingChoiceWorldSession):
    protocol_name = 'nate_adaptiveTimeoutChoiceWorld'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _instantiate_state_machine(self, trial_number=None):
        return AdaptiveTimeoutStateMachine(self.bpod)


if __name__ == '__main__':  # pragma: no cover
    kwargs = get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()
