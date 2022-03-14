import logging
from collections import OrderedDict
import numpy as np
from one.alf.io import AlfBunch

from ibllib.pipes import tasks
import ibllib.io.extractors.training_trials as tt
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
from ibllib.pipes.ephys_preprocessing import (
    EphysRegisterRaw, EphysPulses, RawEphysQC, EphysAudio, EphysMtscomp, EphysVideoCompress, EphysVideoSyncQc,
    EphysCellsQc, EphysDLC, SpikeSorting)
import ibllib.io.extractors.base
import ibllib.io.raw_data_loaders as rawio
from ibllib.qc.task_extractors import TaskQCExtractor

_logger = logging.getLogger('ibllib')


class EphysBanditTrials(tasks.Task):
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

        # TODO

        # trials, files_trials = extract_all(
        #     session_path, bpod_trials=bpod_trials, settings=settings, save=save)
#
        # trials, wheel, output_files = bpod_trials.extract_all(self.session_path, save=True)


class EphysBanditQC(TaskQCExtractor):
    # TODO




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

    base = [tt.RepNum, tt.GoCueTriggerTimes, tt.StimOnTriggerTimes, tt.ItiInTimes, tt.StimOffTriggerTimes,
            tt.StimFreezeTriggerTimes, tt.ErrorCueTriggerTimes, LaserStimulation, ProbabilityRewardLeft, BanditTrialsTable]

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
                tt.FeedbackTimes, tt.FeedbackType, BanditRewardVolume, BanditProbabilityLeft, tt.Wheel]
        exclude = [
            'stimOff_times', 'stimFreeze_times', 'wheel_timestamps', 'wheel_position',
            'wheel_moves_intervals', 'wheel_moves_peak_amplitude', 'peakVelocity_times', 'is_final_movement'
        ]

        out, _ = run_extractor_classes(base, session_path=self.session_path, bpod_trials=self.bpod_trials, settings=self.settings,
                                       save=False)
        table = AlfBunch({k: v for k, v in out.items() if k not in exclude})
        assert len(table.keys()) == 12

        return table.to_df(), *(out.pop(x) for x in self.var_names if x != 'table')


class BanditRewardVolume(tt.RewardVolume):
    """
    Load reward volume delivered for each trial. For trials where the reward was given by laser stimulation
    rather than water stimulation set reward volume to 0
    """

    def _extract(self):
        rewards = super()
        laser = np.array([t['opto_block'] for t in self.bpod_trials]).astype(bool)
        rewards[laser] = 0

        return rewards


class LaserStimulation(BaseBpodTrialsExtractor):
    """
    Get the trials where laser reward stimulation was given. Laser stimulation given when task was in laser block and feedback
    is correct
    """

    save_names = '_ibl_trials.laserStimulation.npy'
    var_names = 'laserStimulation'

    def _extract(self):
        reward = np.array([~np.isnan(t['behavior_data']['States timestamps']['reward'][0][0]) for t in
                           self.bpod_trials]).astype(bool)
        laser = np.array([t['opto_block'] for t in self.bpod_trials]).astype(int)
        laser[~reward] = 0
        return laser


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
        return np.array([self._extract_choice(t).astype(int) for t in self.bpod_trials])

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
        return  np.ones(len(self.bpod_trials)) * np.nan


class BanditContrastLR(tt.ContrastLR):
    """
    Get left and right contrasts from raw datafile. Optionally, saves
    _ibl_trials.contrastLeft.npy and _ibl_trials.contrastRight.npy to alf folder.

    Uses signed_contrast to create left and right contrast vectors.
    """

    def _extract(self):
        contrastLeft = np.array([t['contrast']['value'] for t in self.bpod_trials])
        contrastRight = np.array([t['contrast']['value'] for t in self.bpod_trials])

        return contrastLeft, contrastRight


class EphysBanditPipeline(tasks.Pipeline):
    label = __name__
    def __init__(self, session_path=None, **kwargs):
        super(EphysBanditPipeline, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks["EphysRegisterRaw"] = EphysRegisterRaw(self.session_path)
        tasks["EphysPulses"] = EphysPulses(self.session_path)
        tasks["EphysRawQC"] = RawEphysQC(self.session_path)
        tasks["EphysAudio"] = EphysAudio(self.session_path)
        tasks["EphysMtscomp"] = EphysMtscomp(self.session_path)
        tasks['EphysVideoCompress'] = EphysVideoCompress(self.session_path)
        # level 1
        tasks["SpikeSorting"] = SpikeSorting(
            self.session_path, parents=[tasks["EphysMtscomp"], tasks["EphysPulses"]])
        tasks["EphysBanditTrials"] = EphysBanditTrials(self.session_path, parents=[tasks["EphysPulses"]])
        # level 2
        tasks["EphysVideoSyncQc"] = EphysVideoSyncQc(
            self.session_path, parents=[tasks["EphysVideoCompress"], tasks["EphysPulses"], tasks["EphysPassiveOptoTrials"]])
        tasks["EphysCellsQc"] = EphysCellsQc(self.session_path, parents=[tasks["SpikeSorting"]])
        tasks["EphysDLC"] = EphysDLC(self.session_path, parents=[tasks["EphysVideoCompress"]])
        self.tasks = tasks

__pipeline__ = EphysBanditPipeline
