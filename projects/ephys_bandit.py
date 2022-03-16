import logging
from collections import OrderedDict
import numpy as np

from ibllib.pipes import tasks
import ibllib.io.extractors.training_trials as tt
import project_extraction.projects.training_bandit as bt
from ibllib.io.extractors.base import run_extractor_classes
from ibllib.io.extractors.base import get_session_extractor_type
import ibllib.pipes.ephys_preprocessing as ephys_tasks
from ibllib.io.extractors.ephys_fpga import FpgaTrials
from project_extraction.projects.training_bandit import extract_all as training_extract_all
from ibllib.io.extractors.ephys_fpga import get_main_probe_sync
from ibllib.io.extractors.base import BaseBpodTrialsExtractor

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
                         ('*trials.laserStimulation.npy', 'alf', True),
                         ('*trials.probabilityRewardLeft', 'alf', True),
                         ('*trials.table.pqt', 'alf', True),
                         ('*wheel.position.npy', 'alf', True),
                         ('*wheel.timestamps.npy', 'alf', True),
                         ('*wheelMoves.intervals.npy', 'alf', True),
                         ('*wheelMoves.peakAmplitude.npy', 'alf', True)]
    }

    def _run(self):
        """
        Extracts an ephys bandit
        """
        dsets, out_files = extract_all(self.session_path, save=True)

        return out_files


class BanditFpgaTrials(FpgaTrials):
    save_names = ('_ibl_trials.intervals_bpod.npy',
                  '_ibl_trials.goCueTrigger_times.npy', None, None, None, None, None, None, None,
                  '_ibl_trials.stimOff_times.npy', None, '_ibl_trials.table.pqt',
                  '_ibl_trials.probabilityRewardLeft', '_ibl_trials.laserStimulation.npy'
                  '_ibl_wheel.timestamps.npy', '_ibl_wheel.position.npy',
                  '_ibl_wheelMoves.intervals.npy', '_ibl_wheelMoves.peakAmplitude.npy')
    var_names = ('intervals_bpod',
                 'goCueTrigger_times', 'stimOnTrigger_times',
                 'stimOffTrigger_times', 'stimFreezeTrigger_times', 'errorCueTrigger_times',
                 'errorCue_times', 'itiIn_times',
                 'stimFreeze_times', 'stimOff_times', 'valveOpen_times', 'table', 'probabilityRewardLeft', 'laserStimulation'
                 'wheel_timestamps', 'wheel_position',
                 'wheelMoves_intervals', 'wheelMoves_peakAmplitude')

    # Fields from bpod extractor that we want to resync to FPGA
    bpod_rsync_fields = ['intervals', 'response_times', 'goCueTrigger_times',
                         'stimOnTrigger_times', 'stimOffTrigger_times',
                         'stimFreezeTrigger_times', 'errorCueTrigger_times']

    # Fields from bpod extractor that we want to save
    bpod_fields = ['feedbackType', 'choice', 'rewardVolume', 'contrastLeft', 'contrastRight', 'probabilityLeft',
                   'intervals_bpod', 'probabilityRewardLeft', 'laserStimulation']

    def _extract_bpod(self, bpod_trials, save=False):
        bpod_trials, _ = _extract_all(self.session_path, save=save, bpod_trials=bpod_trials)
        return bpod_trials


def _extract_all(session_path, save=True, bpod_trials=None, settings=None):
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
    base = [bt.BanditRepNum, tt.GoCueTriggerTimes, tt.StimOnTriggerTimes, tt.ItiInTimes, tt.StimOffTriggerTimes,
            tt.StimFreezeTriggerTimes, tt.ErrorCueTriggerTimes, bt.ProbabilityRewardLeft, BanditLaserStimulation,
            EphysBanditTrialsTable]

    trials, files_trials = run_extractor_classes(
        base, save=save, session_path=session_path, bpod_trials=bpod_trials, settings=settings)

    files_wheel = []
    wheel = OrderedDict({k: trials.pop(k) for k in tuple(trials.keys()) if 'wheel' in k})

    _logger.info('session extracted \n')  # timing info in log

    return trials, wheel, (files_trials + files_wheel) if save else None


class EphysBanditTrialsTable(tt.TrialsTable):
    """
    Extracts the following into a table from Bpod raw data:
        intervals, goCue_times, response_times, choice, stimOn_times, contrastLeft, contrastRight,
        feedback_times, feedbackType, rewardVolume, probabilityLeft, firstMovement_times
    Additionally extracts the following wheel data:
        wheel_timestamps, wheel_position, wheel_moves_intervals, wheel_moves_peak_amplitude
    """

    def _extract(self, **kwargs):
        base = [tt.Intervals, tt.GoCueTimes, tt.ResponseTimes, bt.BanditChoice, tt.StimOnOffFreezeTimes, bt.BanditContrastLR,
                tt.FeedbackTimes, tt.FeedbackType, BanditRewardVolume, bt.BanditProbabilityLeft, tt.Wheel]
        exclude = [
            'stimOff_times', 'stimFreeze_times', 'wheel_timestamps', 'wheel_position',
            'wheel_moves_intervals', 'wheel_moves_peak_amplitude', 'peakVelocity_times', 'is_final_movement'
        ]

        out, _ = run_extractor_classes(base, session_path=self.session_path, bpod_trials=self.bpod_trials, settings=self.settings,
                                       save=False)
        table = AlfBunch({k: v for k, v in out.items() if k not in exclude})
        assert len(table.keys()) == 12

        return table.to_df(), *(out.pop(x) for x in self.var_names if x != 'table')


def extract_all(session_path, save=True, bin_exists=False):
    """
    For the IBL ephys task, reads ephys binary file and extract:
        -   sync
        -   wheel
        -   behaviour
        -   video time stamps
    :param session_path: '/path/to/subject/yyyy-mm-dd/001'
    :param save: Bool, defaults to False
    :return: outputs, files
    """
    extractor_type = get_session_extractor_type(session_path)
    _logger.info(f"Extracting {session_path} as {extractor_type}")
    sync, chmap = get_main_probe_sync(session_path, bin_exists=bin_exists)
    outputs, files = run_extractor_classes(
        BanditFpgaTrials, session_path=session_path, save=save, sync=sync, chmap=chmap)
    return outputs, files


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


class BanditLaserStimulation(BaseBpodTrialsExtractor):
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


class EphysBanditPipeline(tasks.Pipeline):
    label = __name__

    def __init__(self, session_path=None, **kwargs):
        super(EphysBanditPipeline, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks["EphysRegisterRaw"] = ephys_tasks.EphysRegisterRaw(self.session_path)
        tasks["EphysPulses"] = ephys_tasks.EphysPulses(self.session_path)
        tasks["EphysRawQC"] = ephys_tasks.RawEphysQC(self.session_path)
        tasks["EphysAudio"] = ephys_tasks.EphysAudio(self.session_path)
        tasks["EphysMtscomp"] = ephys_tasks.EphysMtscomp(self.session_path)
        tasks['EphysVideoCompress'] = ephys_tasks.EphysVideoCompress(self.session_path)
        # level 1
        tasks["SpikeSorting"] = ephys_tasks.SpikeSorting(
            self.session_path, parents=[tasks["EphysMtscomp"], tasks["EphysPulses"]])
        tasks["EphysBanditTrials"] = EphysBanditTrials(self.session_path, parents=[tasks["EphysPulses"]])
        # level 2
        tasks["EphysVideoSyncQc"] = ephys_tasks.EphysVideoSyncQc(
            self.session_path, parents=[tasks["EphysVideoCompress"], tasks["EphysPulses"], tasks["EphysBanditTrials"]])
        tasks["EphysCellsQc"] = ephys_tasks.EphysCellsQc(self.session_path, parents=[tasks["SpikeSorting"]])
        tasks["EphysDLC"] = ephys_tasks.EphysDLC(self.session_path, parents=[tasks["EphysVideoCompress"]])
        self.tasks = tasks
        tasks["EphysPostDLC"] = ephys_tasks.EphysPostDLC(self.session_path,
                                                         parents=[tasks["EphysDLC"], tasks["EphysBanditTrials"],
                                                                  tasks["EphysVideoSyncQc"]])


__pipeline__ = EphysBanditPipeline
