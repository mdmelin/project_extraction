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
from ibllib.io.extractors.ephys_fpga import FpgaTrials
from project_extraction.projects.training_bandit import extract_all as training_extract_all

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


class BanditFpgaTrials(FpgaTrials):
    save_names = ('_ibl_trials.intervals_bpod.npy',
                  '_ibl_trials.goCueTrigger_times.npy', None, None, None, None, None, None, None,
                  '_ibl_trials.stimOff_times.npy', None,
                  '_ibl_wheel.timestamps.npy',
                  '_ibl_wheel.position.npy', '_ibl_wheelMoves.intervals.npy',
                  '_ibl_wheelMoves.peakAmplitude.npy', '_ibl_trials.table.pqt', '_ibl_trials.probabilityRewardLeft',
                  '_ibl_trials.laserStimulation.npy')
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
    bpod_fields = ['feedbackType', 'choice', 'rewardVolume', 'contrastLeft', 'contrastRight', 'probabilityLeft', 'intervals_bpod',
                   'laserStimulation', 'probabilityRewardLeft']

    def _extract_bpod(self, bpod_trials, save=False):
        bpod_trials, _ = training_extract_all(self.session_path, save=save, bpod_trials=bpod_trials)
        return bpod_trials


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
    extractor_type = extractors_base.get_session_extractor_type(session_path)
    _logger.info(f"Extracting {session_path} as {extractor_type}")
    sync, chmap = get_main_probe_sync(session_path, bin_exists=bin_exists)
    outputs, files = extractors_base.run_extractor_classes(
        BanditFpgaTrials, session_path=session_path, save=save, sync=sync, chmap=chmap)
    return outputs, files




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


#check_stimOn_goCue_delays
#check_response_feedback_delays