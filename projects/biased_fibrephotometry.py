"""Extraction pipeline for Alejandro's learning_witten_dop project, task protocol _iblrig_tasks_FPChoiceWorld6.4.2"""
import logging
from collections import OrderedDict

from ibllib.pipes import tasks
from ibllib.qc.base import QC
from ibllib.io.extractors.fibrephotometry import FibrePhotometry as BaseFibrePhotometry
from ibllib.pipes.ephys_preprocessing import (
    EphysRegisterRaw, EphysPulses, RawEphysQC, EphysAudio, EphysMtscomp, EphysVideoCompress, EphysVideoSyncQc,
    EphysCellsQc, EphysDLC, SpikeSorting)
import logging
import warnings
from inspect import getmembers, isfunction

import numpy as np

from ibllib.qc import base
import one.alf.io as alfio
from one.alf.exceptions import ALFObjectNotFound
from one.alf.spec import is_session_path
from iblutil.util import Bunch

_logger = logging.getLogger('ibllib')

# upload to the session endpoint, qc per regions
class FibrePhotometryQC(QC):

    def __init__(self, session_path_or_eid, **kwargs):
        """
        :param session_path_or_eid: A session eid or path
        :param log: A logging.Logger instance, if None the 'ibllib' logger is used
        :param one: An ONE instance for fetching and setting the QC on Alyx
        """
        # When an eid is provided, we will download the required data by default (if necessary)
        download_data = not is_session_path(session_path_or_eid)
        self.download_data = kwargs.pop('download_data', download_data)
        super().__init__(session_path_or_eid, **kwargs)
        self.data = Bunch()

        # QC outcomes map
        self.metrics = None

    def load_data(self, download_data: bool = None) -> None:
        """Extract the data from data files
        Extracts all the required task data from the data files.

        :param download_data: if True, any missing raw data is downloaded via ONE.
        """
        if download_data is not None:
            self.download_data = download_data
        if self.one and not self.one.offline:
            self._ensure_required_data()

        alf_path = self.session_path.joinpath('alf')

        # Load times
        self.data = alfio.load_object(alf_path, '_ibl_photometry')

    def _ensure_required_data(self):
        """
        Ensures the datasets required for QC are local.  If the download_data attribute is True,
        any missing data are downloaded.  If all the data are not present locally at the end of
        it an exception is raised.
        :return:
        """
        for ds in self.dstypes:
            # Check if data available locally
            if not next(self.session_path.rglob(ds), None):
                # If download is allowed, try to download
                if self.download_data is True:
                    assert self.one is not None, 'ONE required to download data'
                    try:
                        self.one.load_dataset(self.eid, ds, download_only=True)
                    except ALFObjectNotFound:
                        raise AssertionError(f'Dataset {ds} not found locally and failed to download')
                else:
                    raise AssertionError(f'Dataset {ds} not found locally and download_data is False')


    def run(self, update: bool = False, **kwargs) -> (str, dict):

        _logger.info(f'Running Fibre Photometry QC for session {self.eid}')
        if all(x is None for x in self.data.values()):
            self.load_data(**kwargs)

        def is_metric(x):
            return isfunction(x) and x.__name__.startswith('check_')

        checks = getmembers(FibrePhotometryQC, is_metric)
        self.metrics = {}
        for k, ch in checks:
            output = ch(self)
            metrics = {f'_roi{i}_' + k[6:]: out for i, out in output}
            self.metrics.update(metrics)


    def check_photobleach(self, n_frames=1000, threshold=0.8):

        qcs = []
        for iD, d in enumerate(self.data['green'].T):
            first = np.mean(d[100:100+n_frames])
            last = np.mean(d[-100+n_frames:-100])
            qcs.append((last / first) > threshold)

        return qcs

    def check_dff_qc(self,):


        # need to do photobleaching correction
        # and then run the qc on this

    def bleach_correct(nacc, avg_window=60, fr=25):
        '''
        Correct for bleaching of gcamp across the session. Calculates
        DF/F
        Parameters
        ----------
        nacc: series with fluorescence data
        avg_window: time for sliding window to calculate F value in seconds
        fr = frame_rate
        '''
        # First calculate sliding window
        avg_window = int(avg_window * fr)
        F = nacc.rolling(avg_window, center=True).mean()
        nacc_corrected = (nacc - F) / F
        return nacc_corrected
















def check_photobleaching_qc(raw_signal, frame_window=1000):
        init = raw_signal[100:1100].mean()
        last = raw_signal[-1100:-100].mean()  # 100 is to avoid init and end artifacts
        qc = 1 * ((last / init) > 0.8)
        return qc

def dff_qc(dff, thres=0.05, frame_interval=40):
        separation_min = 2000 / frame_interval  # 2 seconds separation (10 min)
        peaks = np.where(dff > 0.05)[0]
        qc = 1 * (len(np.where(np.diff(peaks) > separation_min)[0]) > 5)
        return qc


class FibrePhotometry(BaseFibrePhotometry):
    """Unlike base extractor, this uses the Bpod times as the main clock"""

    save_names = ('_ibl_photometry.green.npy', '_ibl_photometry.red.npy',
                  'ibl_photometry.isosbestic.npy', 'ibl_photometry.timestamps.npy')
    var_names = ('green', 'red', 'isobestic', 'timestamps')

    def __init__(self, *args, **kwargs):
        """An extractor for all widefield data"""
        super().__init__(*args, **kwargs)

    def _extract(self, **kwargs):
        """

        Parameters
        ----------

        Returns
        -------
        """
        chmap = kwargs.get('chmap', DEFAULT_CHMAP['mcdaq'])
        fp_path = self.session_path.joinpath('raw_fp_data')
        daq_data = raw_daq_loaders.load_daq_tdms(fp_path, chmap)

        # return [out[k] for k in out] + [wheel['timestamps'], wheel['position'],
        #                                 moves['intervals'], moves['peakAmplitude']]

    def sync_timestamps(self):
        ...


class BiasedFibrephotometryPipeline(tasks.Pipeline):
    label = __name__

    def __init__(self, session_path=None, **kwargs):
        super().__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        # TODO
        tasks["EphysRegisterRaw"] = EphysRegisterRaw(self.session_path)
        tasks["EphysPulses"] = EphysPulses(self.session_path)
        tasks["EphysRawQC"] = RawEphysQC(self.session_path)
        tasks["EphysAudio"] = EphysAudio(self.session_path)
        tasks["EphysMtscomp"] = EphysMtscomp(self.session_path)
        tasks['EphysVideoCompress'] = EphysVideoCompress(self.session_path)
        # level 1
        tasks["SpikeSorting"] = SpikeSorting(
            self.session_path, parents=[tasks["EphysMtscomp"], tasks["EphysPulses"]])
        tasks["EphysBanditTrials"] = BiasedFibrephotometryPipeline(self.session_path, parents=[tasks["EphysPulses"]])
        # level 2
        tasks["EphysVideoSyncQc"] = EphysVideoSyncQc(
            self.session_path, parents=[tasks["EphysVideoCompress"], tasks["EphysPulses"], tasks["EphysPassiveOptoTrials"]])
        tasks["EphysCellsQc"] = EphysCellsQc(self.session_path, parents=[tasks["SpikeSorting"]])
        tasks["EphysDLC"] = EphysDLC(self.session_path, parents=[tasks["EphysVideoCompress"]])
        self.tasks = tasks


__pipeline__ = BiasedFibrephotometryPipeline
