"""Extraction pipeline for Alejandro's learning_witten_dop project, task protocol _iblrig_tasks_FPChoiceWorld6.4.2"""
import logging
from collections import OrderedDict

import numpy as np
import pandas as pd

import one.alf.io as alfio

from ibllib.pipes import tasks
from ibllib.io.extractors.fibrephotometry import DEFAULT_CHMAP, FibrePhotometry as BaseFibrePhotometry
from ibllib.io import raw_daq_loaders
from ibllib.qc.base import QC
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

_logger = logging.getLogger('ibllib').getChild(__name__.split('.')[-1])

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

    def _load_fp(self):
        # Load and rename FP files
        fp_data = alfio.load_object(self.session_path.joinpath('raw_fp_data'), 'fpData')
        alfio.load_file_content()
        try:
            fp_data = pd.read_csv(ses + '/alf/fp_data/FP470')
        except:
            fp_data = pd.read_csv(ses + '/alf/fp_data/FP470.csv')

        fp_data = fp_data.rename(columns={'Region2G': 'DMS', 'Region1G': 'NAcc', 'Region0G': 'DLS'})

        try:
            fp_data_415 = pd.read_csv(ses + '/alf/fp_data/FP415')
        except:
            fp_data_415 = pd.read_csv(ses + '/alf/fp_data/FP415.csv')

        # TODO This region / ROI map should be in the Alyx insersions table?
        fp_data_415 = fp_data_415.rename(columns={'Region2G': 'DMS_isos', 'Region1G': 'NAcc_isos', 'Region0G': 'DLS_isos'})

    def sync_timestamps(self, daq_data, fp_data, trials_data):
        """

        Parameters
        ----------
        daq_data : dict
            Dictionary with keys ('bpod', 'fp') containing voltage values acquired from the DAQ
        fp_data : pandas.DataFrame
            The raw fibrephotometry data
        trials_data : dict
            The Bpod trial events

        Returns
        -------

        """
        daq_data = pd.DataFrame.from_dict(daq_data)
        # Threshold Convert analogue
        daq_data['fp'] = 1 * (daq_data['fp'] >= 4)
        daq_data['bpod'] = 1 * (daq_data['bpod'] >= 2)

        # Patch session if needed: Delete short pulses (sample smaller than frame acquisition rate) or
        # pulses before acquistion for FP and big breaks (acquisition started twice)
        daq_data.loc[np.where(daq_data['fp'].diff() == 1)[0], 'TTL_change'] = 1
        sample_ITI = np.median(np.diff(daq_data.loc[daq_data['TTL_change'] == 1].index))
        if sample_ITI == 10:  # New protocol saves ITI for all: 470,145 and 2x empty frames
            true_FP = daq_data.loc[daq_data['TTL_change'] == 1].index[::4]
            daq_data['TTL_change'] = 0
            daq_data['fp'] = 0
            daq_data.iloc[true_FP, daq_data.columns.get_loc('TTL_change')] = 1
            daq_data.iloc[true_FP, daq_data.columns.get_loc('fp')] = 1
            daq_data.iloc[true_FP + 1, daq_data.columns.get_loc('fp')] = 1  # Pulses are 2ms long
            daq_data.loc[np.where(daq_data['fp'].diff() == 1)[0], 'TTL_change'] = 1
            sample_ITI = np.median(np.diff(daq_data.loc[daq_data['TTL_change'] == 1].index))
        print(sample_ITI)
        while np.diff(daq_data.loc[daq_data['TTL_change'] == 1].index).max() > sample_ITI * 4:  # Session was started twice
            print('Session started twice')
            ttl_id = np.where(np.diff(daq_data.loc[daq_data['TTL_change'] == 1].index) == np.diff(
                daq_data.loc[daq_data['TTL_change'] == 1].index).max())[0][0]
            real_id = daq_data.loc[daq_data['TTL_change'] == 1].index[ttl_id]
            daq_data.iloc[:int(real_id + np.diff(daq_data.loc[daq_data['TTL_change'] == 1].index).max() - sample_ITI), :] = 0

        pulse_to_del = daq_data.loc[daq_data['TTL_change'] == 1].index[np.where(
            (np.diff(daq_data.loc[daq_data['TTL_change'] == 1].index) < sample_ITI * 0.95) | (
                        np.diff(daq_data.loc[daq_data['TTL_change'] == 1].index) > sample_ITI * 1.05))[0]]
        for i in pulse_to_del:
            print(len(pulse_to_del) + "noise frames")
            daq_data.iloc[i:int(i + sample_ITI * 1.05), np.where(daq_data.columns == 'fp')[0]] = 0
        # Update TTL change column
        daq_data['TTL_change'] = 0
        daq_data.loc[np.where(daq_data['fp'].diff() == 1)[0], 'TTL_change'] = 1
        # Check that there aren't too many empty frames
        assert abs(len(np.where(daq_data['fp'].diff() == 1)[0]) - len(fp_data)) < 6

        # Align events
        fp_data['DAQ_timestamp'] = np.nan
        daq_idx = fp_data.columns.get_loc('DAQ_timestamp')
        fp_data.iloc[:, daq_idx] = \
            np.where(daq_data['fp'].diff() == 1)[0][:len(fp_data)]

        # Extract Trial Events
        daq_data.loc[np.where(daq_data['bpod'].diff() == 1)[0], 'bpod_on'] = 1
        daq_data.loc[np.where(daq_data['bpod'].diff() == -1)[0], 'bpod_off'] = 1
        daq_data.loc[np.where(daq_data['bpod'].diff() == 1)[0], 'bpod_duration'] = \
            daq_data.loc[daq_data['bpod_off'] == 1].index - \
            daq_data.loc[daq_data['bpod_on'] == 1].index
        daq_data['feedbackTimes'] = np.nan
        daq_data.loc[daq_data['bpod_duration'] > 100, 'feedbackTimes'] = 1
        daq_data['bpod_event'] = np.nan
        daq_data.loc[daq_data['bpod_duration'] > 1000, 'bpod_event'] = 'error'
        daq_data.loc[daq_data['bpod_duration'] <= 100, 'bpod_event'] = 'trial_start'
        daq_data.loc[(daq_data['bpod_duration'] > 100) &
                   (daq_data['bpod_duration'] < 1000), 'bpod_event'] = 'reward'

        # Interpolate times from bpod clock
        assert (len(daq_data.loc[daq_data['feedbackTimes'] == 1]) - len(trials_data['feedback_times'])) <= 1
        daq_data['bpod_time'] = np.nan
        nan_trials = np.where(trials_data['choice'] == 0)[0]  # No choice was made
        if len(nan_trials) != 0:
                try:  # For new code with bpod pulses also in NO GOs
                    daq_data.loc[daq_data['feedbackTimes'] == 1, 'bpod_time'] = trials_data['feeback_times']
                except:  # For older code without pulses in nan
                    trials_data['feeback_times'] = np.delete(trials_data['feeback_times'], nan_trials)
        if (len(daq_data.loc[daq_data['feedbackTimes'] == 1]) - len(trials_data['feeback_times'])) == 0:
            daq_data.loc[daq_data['feedbackTimes'] == 1, 'bpod_time'] = trials_data['feeback_times']
        if (len(daq_data.loc[daq_data['feedbackTimes'] == 1]) - len(trials_data['feeback_times'])) == 1:  # If bpod didn't save last trial
            print('Bpod missing last trial')
            daq_data.loc[daq_data['feedbackTimes'] == 1, 'bpod_time'] = np.append(trials_data['feeback_times'], np.nan)
        daq_data['bpod_time'].interpolate(inplace=True)
        # Delete values after last known bpod - interpolate will not extrapolate!
        daq_data.iloc[np.where(daq_data['bpod_time'] == daq_data['bpod_time'].max())[0][1]:,
                    daq_data.columns.get_loc('bpod_time')] = np.nan
        # Align fluo data with bpod time
        fp_data['bpod_time'] = np.nan
        daq_idx = fp_data.columns.get_loc('bpod_time')
        fp_data.iloc[:, daq_idx] = \
            daq_data['bpod_time'].to_numpy()[np.where(daq_data['fp'].diff() == 1)[0][:len(fp_data)]]

        # Save data
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
