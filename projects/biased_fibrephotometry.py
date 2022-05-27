"""Extraction pipeline for Alejandro's learning_witten_dop project, task protocol _iblrig_tasks_FPChoiceWorld6.4.2"""
import logging
from inspect import getmembers, isfunction
from collections import OrderedDict

import numpy as np
import pandas as pd
import one.alf.io as alfio
from one.alf.exceptions import ALFObjectNotFound
from one.alf.spec import is_session_path
from iblutil.util import Bunch

from ibllib.pipes import tasks
from ibllib.io.extractors.fibrephotometry import FibrePhotometry as BaseFibrePhotometry
from ibllib.pipes.fibrephotometry import FibrePhotometryPreprocess as BaseFibrePhotometryPreprocess
from ibllib.io import raw_daq_loaders
from neurodsp.utils import sync_timestamps
from ibllib.qc.base import QC
from ibllib.pipes.training_preprocessing import (
    TrainingRegisterRaw, TrainingAudio, TrainingTrials, TrainingDLC, TrainingStatus, TrainingVideoCompress)


_logger = logging.getLogger('ibllib').getChild(__name__.split('.')[-1])

# MCC USB-201
DEFAULT_CHMAP = {
    'mccdaq': {
        'bpod': 'ai1',
        'fp': 'ai0'  # Fibrephotometry
    },
}


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
        ...

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
    """Unlike base extractor, this uses the Bpod times as the main clock and syncs to MCC DAQ"""

    def __init__(self, *args, **kwargs):
        """An extractor for all biased_fibrephotometry task data"""
        super().__init__(*args, **kwargs)

    def extract_timestamps(self, fp_data, chmap=None, **kwargs):
        """Extract and sync the fibrephotometry timestamps acquired through the MCC DAQ.

        This function is called by the _extract method of the super class.

        Parameters
        ----------
        fp_data : dict
            A Bunch of raw fibrephotometry data, with the keys ('raw', 'channels').
        chmap : dict
            An optional dict of DAQ channel IDs for 'bpod' and 'fp' channel inputs.

        Returns
        -------
        numpy.ndarray
            A 1D array of timestamps, one per frame
        """
        chmap = kwargs.get('chmap', DEFAULT_CHMAP['mccdaq'])
        daq_data = raw_daq_loaders.load_daq_tdms(self.session_path / 'raw_photometry_data', chmap)
        trials_data = alfio.load_object(self.session_path / 'alf', 'trials')
        return self.sync_timestamps(daq_data, fp_data, trials_data)

    def sync_timestamps(self, daq_data, fp_data, trials_data):
        """
        Converts the Neurophotometrics frame timestamps in Bpod time using the Bpod feedback times.

        Both the Neurophotometrics and Bpod devices are connected to a common DAQ. The Bpod sends
        a TTL to the DAQ when feedback is delivered (among other things).  The Neurophotometrics
        sends TTLs when a frame in the 470 channel is acquired.  This DAQ only records samples
        (not timestamps) so the Bpod clock is used.  First the feedback TTLs are identified on the
        DAQ based on length. The Bpod times for these TTLs are taken from the trials data and
        assigned to a DAQ sample.  Times for the other DAQ samples are calculated using linear
        interpolation.  Those timestamps for the samples where a Neurophotometrics TTL was received
        are sync'd to the raw Neurophotometrics clock timestamps.  Using the determined
        interpolation function, the raw Neurophotometrics timestamps for each frame are converted
        to Bpod time.

        Parameters
        ----------
        daq_data : dict
            Dictionary with keys ('bpod', 'fp') containing voltage values acquired from the DAQ
        fp_data : dict
            A Bunch containing the raw fibrephotometry data output from Neurophotometrics Bonsai
            workflow and a table of channel values for interpreting the frame state.
        trials_data : dict
            An ALF trials object containing Bpod trials events.  Only `feedback_times` and `choice`
            keys are required

        Returns
        -------
        numpy.ndarray
            An array of frame timestamps the length of fp_data
        """
        # For extraction we only use 470 channel - this channel is accompanied with a TTL
        # Column name 'Flags' in older Bonsai workflow
        fp_channels = fp_data['channels']
        fp_data = fp_data['raw'].copy()
        state = fp_data.get('LedState', fp_data.get('Flags', None))
        mask = state.isin(fp_channels['L470'])
        frame_numbers_470 = fp_data['FrameCounter'][mask]

        daq_data = pd.DataFrame.from_dict(daq_data)
        # Threshold Convert analogue
        daq_data['fp'] = 1 * (daq_data['fp'] >= 4)
        daq_data['bpod'] = 1 * (daq_data['bpod'] >= 2)

        # Patch session if needed: Delete short pulses (sample smaller than frame acquisition rate) or
        # pulses before acquisition for FP and big breaks (acquisition started twice)
        daq_data.loc[np.where(daq_data['fp'].diff() == 1)[0], 'TTL_change'] = 1
        pulse_interval = np.median(np.diff(daq_data.loc[daq_data['TTL_change'] == 1].index))
        # Fix for when TTL send on every single frame instead of 470 channel frame
        # 10 because acquisition was at 25Hz; DAQ - 1000Hz
        # Frame rate from FP data file should match frame rate on DAQ
        if pulse_interval == 10:  # New protocol saves ITI for all: 470,145 and 2x empty frames
            _logger.debug('TTL on every frame; correcting (removing pulses from non-470 channels)')
            true_FP = daq_data.loc[daq_data['TTL_change'] == 1].index[::4]
            daq_data['TTL_change'] = 0
            daq_data['fp'] = 0
            daq_data.iloc[true_FP, daq_data.columns.get_loc('TTL_change')] = 1
            daq_data.iloc[true_FP, daq_data.columns.get_loc('fp')] = 1
            daq_data.iloc[true_FP + 1, daq_data.columns.get_loc('fp')] = 1  # Pulses are 2ms long
            daq_data.loc[np.where(daq_data['fp'].diff() == 1)[0], 'TTL_change'] = 1
            # Double-check fix worked, get corrected interval
            pulse_interval = np.median(np.diff(daq_data.loc[daq_data['TTL_change'] == 1].index))
        _logger.info(f'Pulse interval: {pulse_interval}')

        # Bpod crashed and was restarted but DAQ wasn't restarted
        # If greater than 4 times the 410 chan frame rate, there must have been a long gap
        # Do this until no more gaps
        while np.diff(daq_data.loc[daq_data['TTL_change'] == 1].index).max() > pulse_interval * 4:
            _logger.info('Large gaps detected, assuming false starts; removing...')
            # Find the biggest TTL gap index
            ttl_id = np.where(np.diff(daq_data.loc[daq_data['TTL_change'] == 1].index) ==
                np.diff(daq_data.loc[daq_data['TTL_change'] == 1].index).max())[0][0]
            # Transform into actual DAQ index
            real_id = daq_data.loc[daq_data['TTL_change'] == 1].index[ttl_id]
            # Remove all TTLs up to this large gap (set to 0)
            daq_data.iloc[:int(real_id + np.diff(daq_data.loc[daq_data['TTL_change'] == 1].index).max() - pulse_interval), :] = 0

        # Update TTL change column
        daq_data['TTL_change'] = 0
        daq_data.loc[np.where(daq_data['fp'].diff() == 1)[0], 'TTL_change'] = 1

        # Check that there aren't too many empty frames
        # Number of recorded TTLs on DAQ should roughly equal the number of FP 470 frames
        # Six was empirically determined: There is a delay between Bonsai stopping a session and the
        # LED (sending the TTLs) permanently turning off. In a recent test I saw that if I acquire at
        # 15Hz there is no extra pulses, at 25Hz 2 pulses and at 100Hz 5 pulses.
        dropped_frames = abs(len(np.where(daq_data['fp'].diff() == 1)[0]) - len(frame_numbers_470))
        assert dropped_frames < 6
        _logger.debug(f'{dropped_frames} dropped frames')

        # Align events
        fp_data['DAQ_idx'] = np.nan
        daq_idx = fp_data.columns.get_loc('DAQ_idx')
        daq_ttl_edges = np.where(daq_data['fp'].diff() == 1)[0]
        # discard DAQ TTLs missed at end of session
        fp_data.iloc[frame_numbers_470, daq_idx] = daq_ttl_edges[:len(frame_numbers_470)]

        # Extract Trial Events
        daq_data.loc[np.where(daq_data['bpod'].diff() == 1)[0], 'bpod_on'] = 1
        daq_data.loc[np.where(daq_data['bpod'].diff() == -1)[0], 'bpod_off'] = 1
        daq_data.loc[np.where(daq_data['bpod'].diff() == 1)[0], 'bpod_duration'] = \
            daq_data.loc[daq_data['bpod_off'] == 1].index - \
            daq_data.loc[daq_data['bpod_on'] == 1].index
        daq_data['feedbackTimes'] = np.nan
        daq_data.loc[daq_data['bpod_duration'] > 105, 'feedbackTimes'] = 1
        daq_data['bpod_event'] = np.nan
        daq_data.loc[daq_data['bpod_duration'] > 1000, 'bpod_event'] = 'error'
        daq_data.loc[daq_data['bpod_duration'] <= 105, 'bpod_event'] = 'trial_start'
        # Changes from rig-to-rig depending on valve calibration
        daq_data.loc[(daq_data['bpod_duration'] > 100) &
                   (daq_data['bpod_duration'] < 1000), 'bpod_event'] = 'reward'

        # Interpolate times from bpod clock
        # Check we have same number of trial on DAQ and Bpod
        assert abs(len(daq_data.loc[daq_data['feedbackTimes'] == 1]) - len(trials_data['feedback_times'])) <= 1

        daq_data['bpod_time'] = np.nan
        nan_trials = np.where(trials_data['choice'] == 0)[0]  # No choice was made
        if len(nan_trials) != 0:  # If there are no gos
            try:  # For new code with bpod pulses also in NO GOs
                # Assign Bpod feedback times to each TTL sample
                daq_data.loc[daq_data['feedbackTimes'] == 1, 'bpod_time'] = trials_data['feeback_times']
            except:  # For older code without pulses in nan
                if len(trials_data['feedback_times']) > len(daq_data.loc[daq_data['feedbackTimes'] == 1]):
                    trials_data['feedback_times'] = np.delete(trials_data['feedback_times'], nan_trials)

        # If number of pulses match, assign trials.feedback_times to 'bpod_time' column.
        # We will interpolate these times for each frame next
        if (len(daq_data.loc[daq_data['feedbackTimes'] == 1]) - len(trials_data['feedback_times'])) == 0:
            daq_data.loc[daq_data['feedbackTimes'] == 1, 'bpod_time'] = trials_data['feedback_times']

        # If task was stopped before last trial ended, add nan for last pulse
        if (len(daq_data.loc[daq_data['feedbackTimes'] == 1]) - len(trials_data['feedback_times'])) == 1:
            _logger.debug('Bpod missing last trial')
            daq_data.loc[daq_data['feedbackTimes'] == 1, 'bpod_time'] = np.append(trials_data['feedback_times'], np.nan)
        # TODO Assert feedback TTLs match feedback times

        # Interpolate Bpod times
        daq_data['bpod_time'].interpolate(inplace=True)
        # Delete values after last known bpod - interpolate will not extrapolate!
        daq_data.iloc[np.where(daq_data['bpod_time'] == daq_data['bpod_time'].max())[0][1]:,
                    daq_data.columns.get_loc('bpod_time')] = np.nan
        # Align photometry data with bpod time
        fp_data['bpod_time'] = np.nan
        daq_idx = fp_data.columns.get_loc('bpod_time')
        # Assign Bpod time for all photometry DAQ pulses to new column to later save into timestamps ALF
        fp_data.iloc[frame_numbers_470, daq_idx] = \
            daq_data['bpod_time'].to_numpy()[daq_ttl_edges[:len(frame_numbers_470)]]

        # Get the interpolated Bpod timestamps and FP clock timestamps
        fp_daq_ts = fp_data.iloc[frame_numbers_470, daq_idx]
        fp_daq_ts = fp_daq_ts[~fp_daq_ts.isna()]
        fp_ts = fp_data['Timestamp'].iloc[fp_daq_ts.index].values
        # Try to sync the two
        _logger.info('Synching Neurophotometrics clock to Bpod timestamps...')
        fcn, drift = sync_timestamps(fp_ts, fp_daq_ts.values)
        _logger.info(f'Drift: {drift:.2f}')

        # Using the interpolation function from sync_timestamps, convert FP clock timestamps to Bpod ones
        fp_data.iloc[:, daq_idx] = fcn(fp_data['Timestamp'].values)
        assert fp_data['bpod_time'].is_monotonic_increasing

        return fp_data['bpod_time'].values


class FibrePhotometryPreprocess(BaseFibrePhotometryPreprocess):

    def _run(self, **kwargs):
        _, out_files = FibrePhotometry(self.session_path).extract(save=True)
        return out_files


class BiasedFibrephotometryPipeline(tasks.Pipeline):
    label = __name__

    def __init__(self, session_path=None, **kwargs):
        super().__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks['TrainingRegisterRaw'] = TrainingRegisterRaw(self.session_path)
        tasks['TrainingTrials'] = TrainingTrials(self.session_path)
        tasks['TrainingVideoCompress'] = TrainingVideoCompress(self.session_path)
        tasks['TrainingAudio'] = TrainingAudio(self.session_path)
        # level 1
        tasks['BiasedFibrePhotometry'] = FibrePhotometryPreprocess(self.session_path, parents=[tasks['TrainingTrials']])
        tasks['TrainingStatus'] = TrainingStatus(self.session_path, parents=[tasks['TrainingTrials']])
        tasks['TrainingDLC'] = TrainingDLC(
            self.session_path, parents=[tasks['TrainingVideoCompress']])
        self.tasks = tasks


__pipeline__ = BiasedFibrephotometryPipeline
