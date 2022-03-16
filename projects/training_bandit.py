import logging
from collections import OrderedDict
import numpy as np
from one.alf.io import AlfBunch

from ibllib.pipes import tasks
import ibllib.io.extractors.training_trials as tt
import ibllib.pipes.training_preprocessing as training_tasks
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
import ibllib.io.extractors.base
import ibllib.io.raw_data_loaders as rawio


_logger = logging.getLogger('ibllib')


class TrainingBanditTrials(tasks.Task):
    priority = 90
    level = 0
    force = False
    signature = {
        'input_files': [('_iblrig_taskData.raw.*', 'raw_behavior_data', True),
                        ('_iblrig_taskSettings.raw.*', 'raw_behavior_data', True),
                        ('_iblrig_encoderEvents.raw*', 'raw_behavior_data', True),
                        ('_iblrig_encoderPositions.raw*', 'raw_behavior_data', True)],
        'output_files': [('*trials.goCueTrigger_times.npy', 'alf', True),
                         ('*trials.itiDuration.npy', 'alf', False),
                         ('*trials.probabilityRewardLeft', 'alf', True),
                         ('*trials.table.pqt', 'alf', True),
                         ('*wheel.position.npy', 'alf', True),
                         ('*wheel.timestamps.npy', 'alf', True),
                         ('*wheelMoves.intervals.npy', 'alf', True),
                         ('*wheelMoves.peakAmplitude.npy', 'alf', True)]
    }

    def _run(self):
        """
        Extracts an iblrig training session
        """
        trials, wheel, output_files = extract_all(self.session_path, save=True)
        if trials is None:
            return None

        return output_files


def extract_all(session_path, save=True, bpod_trials=None, settings=None):
    """Extract trials and wheel data.

    For task versions >= 5.0.0, outputs wheel data and trials.table dataset (+ some extra datasets)

    Parameters
    ----------
    session_path : str, pathlib.Path
        The path to the session
    save : bool
        If true save the data files to ALF
    bpod_trials : list of dicts
        The Bpod trial dicts loaded from the _iblrig_taskData.raw dataset
    settings : dict
        The Bpod settings loaded from the _iblrig_taskSettings.raw dataset

    Returns
    -------
    A list of extracted data and a list of file paths if save is True (otherwise None)
    """

    extractor_type = ibllib.io.extractors.base.get_session_extractor_type(session_path)
    _logger.info(f"Extracting {session_path} as {extractor_type}")
    bpod_trials = bpod_trials or rawio.load_data(session_path)
    settings = settings or rawio.load_settings(session_path)
    _logger.info(f'{extractor_type} session on {settings["PYBPOD_BOARD"]}')

    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    # check that the extraction works for both the shaping 0-100 and the other one
    base = [BanditRepNum, tt.GoCueTriggerTimes, tt.StimOnTriggerTimes, tt.ItiInTimes, tt.StimOffTriggerTimes,
            tt.StimFreezeTriggerTimes, tt.ErrorCueTriggerTimes, ProbabilityRewardLeft, BanditTrialsTable]

    trials, files_trials = run_extractor_classes(
        base, save=save, session_path=session_path, bpod_trials=bpod_trials, settings=settings)

    files_wheel = []
    wheel = OrderedDict({k: trials.pop(k) for k in tuple(trials.keys()) if 'wheel' in k})

    _logger.info('session extracted \n')  # timing info in log

    return trials, wheel, (files_trials + files_wheel) if save else None


class BanditTrialsTable(tt.TrialsTable):
    """
    Extracts the following into a table from Bpod raw data:
        intervals, goCue_times, response_times, choice, stimOn_times, contrastLeft, contrastRight,
        feedback_times, feedbackType, rewardVolume, probabilityLeft, firstMovement_times
    Additionally extracts the following wheel data:
        wheel_timestamps, wheel_position, wheel_moves_intervals, wheel_moves_peak_amplitude
    """

    def _extract(self, **kwargs):
        base = [tt.Intervals, tt.GoCueTimes, tt.ResponseTimes, BanditChoice, tt.StimOnOffFreezeTimes, BanditContrastLR,
                tt.FeedbackTimes, tt.FeedbackType, tt.RewardVolume, BanditProbabilityLeft, tt.Wheel]
        exclude = [
            'stimOff_times', 'stimFreeze_times', 'wheel_timestamps', 'wheel_position',
            'wheel_moves_intervals', 'wheel_moves_peak_amplitude', 'peakVelocity_times', 'is_final_movement'
        ]

        out, _ = run_extractor_classes(base, session_path=self.session_path, bpod_trials=self.bpod_trials, settings=self.settings,
                                       save=False)
        table = AlfBunch({k: v for k, v in out.items() if k not in exclude})
        assert len(table.keys()) == 12

        return table.to_df(), *(out.pop(x) for x in self.var_names if x != 'table')


class BanditRepNum(tt.RepNum):

    def _extract(self):
        repnum = np.ones(len(self.bpod_trials))

        return repnum


class BanditChoice(tt.Choice):
    """
    Get the subject's choice in every trial.
    **Optional:** saves _ibl_trials.choice.npy to alf folder.

    Extracts choice from side on which first rotary encoder threhsold was detected
    -1 is a CCW turn (towards the left) this is stored in 'RotaryEncoder1_1'
    +1 is a CW turn (towards the right) this is stored in 'RotaryEncoder1_2'
    0 is a no_go trial

    """

    def _extract(self):
        return np.array([self._extract_choice(t) for t in self.bpod_trials]).astype(int)

    def _extract_choice(self, data):
        if (('RotaryEncoder1_2' in data['behavior_data']['Events timestamps']) &
                ('RotaryEncoder1_1' in data['behavior_data']['Events timestamps'])):
            # When both thresholds are passed, take the first crossed threshold to be the choice direction
            choice = -1 if (data['behavior_data']['Events timestamps']['RotaryEncoder1_2'][0] >
                            data['behavior_data']['Events timestamps']['RotaryEncoder1_1'][0]) else 1
        elif 'RotaryEncoder1_1' in data['behavior_data']['Events timestamps']: # this should be CCW
            choice = -1
        elif 'RotaryEncoder1_2' in data['behavior_data']['Events timestamps']: # this should be CW
            choice = 1
        else:
            choice = 0

        return choice


class ProbabilityRewardLeft(BaseBpodTrialsExtractor):
    """
    Probability of reward being given on left movement of wheel
    """
    save_names = '_ibl_trials.probabilityRewardLeft.npy'
    var_names = 'probabilityRewardLeft'

    def _extract(self, **kwargs):
        return np.array([t['stim_probability_left'] for t in self.bpod_trials])


class BanditProbabilityLeft(tt.ProbabilityLeft):
    """
    No visual stimulus related events so probability left is nan for all trials
    """
    save_names = '_ibl_trials.probabilityLeft.npy'
    var_names = 'probabilityLeft'

    def _extract(self, **kwargs):
        return np.ones(len(self.bpod_trials)) * np.nan


class BanditContrastLR(tt.ContrastLR):
    """
    Get left and right contrasts from raw datafile. Optionally, saves
    _ibl_trials.contrastLeft.npy and _ibl_trials.contrastRight.npy to alf folder.

    Uses signed_contrast to create left and right contrast vectors.
    """

    def _extract(self):
        contrastLeft = np.array([t['contrast'] for t in self.bpod_trials])
        contrastRight = np.array([t['contrast'] for t in self.bpod_trials])

        return contrastLeft, contrastRight



class TrainingBanditPipeline(tasks.Pipeline):
    label = __name__

    def __init__(self, session_path, **kwargs):
        super(TrainingBanditPipeline, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks['TrainingRegisterRaw'] = training_tasks.TrainingRegisterRaw(self.session_path)
        tasks['TrainingBanditTrials'] = TrainingBanditTrials(self.session_path)
        tasks['TrainingVideoCompress'] = training_tasks.TrainingVideoCompress(self.session_path)
        tasks['TrainingAudio'] = training_tasks.TrainingAudio(self.session_path)
        # level 1
        tasks['TrainingDLC'] = training_tasks.TrainingDLC(
            self.session_path, parents=[tasks['TrainingVideoCompress']])
        self.tasks = tasks
