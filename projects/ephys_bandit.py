from collections import OrderedDict

import numpy as np

import one.alf.io as alfio
from ibllib.io.extractors import ephys_fpga
from ibllib.dsp.utils import sync_timestamps
from ibllib.plots import squares, vertical_lines
from ibllib.pipes import tasks

from ibllib.pipes.ephys_preprocessing import (
    EphysRegisterRaw, EphysPulses, RawEphysQC, EphysAudio, EphysMtscomp, EphysVideoCompress, EphysVideoSyncQc,
    EphysCellsQc, EphysDLC, SpikeSorting)


LASER_PULSE_DURATION_SECS = .5
LASER_PROBABILITY = .8
DISPLAY = False

class EphysBanditTrials(tasks.Task):
    # Do some extraction stuff



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
        tasks["EphysPassiveOptoTrials"] = EphysBanditTrials(self.session_path, parents=[tasks["EphysPulses"]])
        # level 2
        tasks["EphysVideoSyncQc"] = EphysVideoSyncQc(
            self.session_path, parents=[tasks["EphysVideoCompress"], tasks["EphysPulses"], tasks["EphysPassiveOptoTrials"]])
        tasks["EphysCellsQc"] = EphysCellsQc(self.session_path, parents=[tasks["SpikeSorting"]])
        tasks["EphysDLC"] = EphysDLC(self.session_path, parents=[tasks["EphysVideoCompress"]])
        self.tasks = tasks
__pipeline__ = EphysBanditPipeline
