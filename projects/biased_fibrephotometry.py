"""Extraction pipeline for Alejandro's learning_witten_dop project, task protocol _iblrig_tasks_FPChoiceWorld6.4.2"""
import logging
from inspect import getmembers, isfunction

import numpy as np
import pandas as pd
import one.alf.io as alfio
from one.alf.exceptions import ALFObjectNotFound
from one.alf.spec import is_session_path
from iblutil.util import Bunch

from ibllib.io.extractors.fibrephotometry import FibrePhotometry as BaseFibrePhotometry
from ibllib.io.extractors.fibrephotometry import DAQ_CHMAP, NEUROPHOTOMETRICS_LED_STATES
from ibllib.pipes.photometry_tasks import FibrePhotometryPreprocess as PhotometryPreprocess
from ibllib.io import raw_daq_loaders
from ibllib.qc.base import QC
from scipy import interpolate

CHANNELS = pd.DataFrame.from_dict(NEUROPHOTOMETRICS_LED_STATES)

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
            first = np.mean(d[100:100 + n_frames])
            last = np.mean(d[-100 + n_frames:-100])
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

    def _extract(self, light_source_map=None, collection=None, regions=None, **kwargs):
        """

        Parameters
        ----------
        regions: list of str
            The list of regions to extract. If None extracts all columns containing "Region". Defaults to None.
        light_source_map : dict
            An optional map of light source wavelengths (nm) used and their corresponding colour name.
        collection: str / pathlib.Path
            An optional relative path from the session root folder to find the raw photometry data.
            Defaults to `raw_photometry_data`

        Returns
        -------
        numpy.ndarray
            A 1D array of signal values.
        numpy.ndarray
            A 1D array of ints corresponding to the active light source during a given frame.
        pandas.DataFrame
            A table of intensity for each region, with associated times, wavelengths, names and colors
        """
        out_df = super()._extract(light_source_map=None, collection=None, regions=None, **kwargs)

        fp_path = self.session_path.joinpath('raw_fp_data')
        if not fp_path.exists():
            fp_path = self.session_path.joinpath('alf', 'fp_data')

        fp_data = alfio.load_object(self.session_path.joinpath(self.collection), 'fpData')['raw']
        processed = pd.read_csv(fp_path.joinpath('FP470_processed.csv'))
        assert all(processed['FrameCounter'].diff()[1:] == processed['FrameCounter'].diff().median())
        include = np.zeros_like(out_df['times'].values, dtype=bool)
        state = fp_data.get('LedState', fp_data.get('Flags', None))
        mask = state.isin(CHANNELS['L470'])
        frame_470 = np.where(mask)[0]
        first_470 = frame_470[0]
        diff_470 = np.diff(frame_470)[0]
        fr_start = np.where(fp_data['FrameCounter'].values == processed.iloc[0]['FrameCounter'])[0][0] - first_470
        fr_end = (np.where(fp_data['FrameCounter'].values == processed.iloc[-1]['FrameCounter'])[0][0] +
                  (diff_470 - first_470))
        include[fr_start:fr_end] = 1
        out_df['include'] = include

        return out_df

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
        chmap = kwargs.get('chmap', DAQ_CHMAP)
        daq_data = raw_daq_loaders.load_channels_tdms(self.session_path.joinpath(self.collection), chmap=chmap)[0]
        trials_data = alfio.load_object(self.session_path / 'alf', 'trials')
        return self.sync_timestamps(daq_data, fp_data, trials_data)

    @staticmethod
    def sync_timestamps(daq_data, fp_data, trials):
        """
        Converts the Neurophotometrics frame timestamps in Bpod time using the Bpod feedback times.

        Both the Neurophotometrics and Bpod devices are connected to a common DAQ. The Bpod sends
        a TTL to the DAQ when feedback is delivered (among other things).  The Neurophotometrics
        sends TTLs when a frame in the 470 channel is acquired.  The Bpod is used as the master
        clock.  First the feedback TTLs are identified on the DAQ based on length. The Bpod times
        for these TTLs are taken from the trials data and assigned to a DAQ sample.  Times for the
        other DAQ samples are calculated using linear interpolation.  Those timestamps for the
        samples where a Neurophotometrics TTL was received are sync'd to the raw Neurophotometrics
        clock timestamps.  Using the determined interpolation function, the raw Neurophotometrics
        timestamps for each frame are converted to Bpod time.

        Parameters
        ----------
        daq_data : dict
            Dictionary with keys ('bpod', 'fp') containing voltage values acquired from the DAQ
        fp_data : dict
            A Bunch containing the raw fibrephotometry data output from Neurophotometrics Bonsai
            workflow and a table of channel values for interpreting the frame state.
        trials : dict
            An ALF trials object containing Bpod trials events.  Only `feedback_times` and `choice`
            keys are required

        Returns
        -------
        numpy.ndarray
            An array of frame timestamps the length of fp_data
        """

        daq_data = pd.DataFrame.from_dict(daq_data)
        daq_data['photometry'] = 1 * (daq_data['photometry'] >= 4)
        daq_data['bpod'] = 1 * (daq_data['bpod'] >= 2)

        # Find if the session has been started more than twice
        daq_data.loc[np.where(daq_data['photometry'].diff() == 1)[0], 'photometry_ttl'] = 1
        ttl_interval = np.median(np.diff(daq_data.loc[daq_data['photometry_ttl'] == 1].index))
        state = fp_data.get('LedState', fp_data.get('Flags', None))
        mask = state.isin(CHANNELS['L470'])
        if ttl_interval == 40:
            frame_number = fp_data['FrameCounter'][mask]
        elif ttl_interval == 20:
            frame_number = fp_data['FrameCounter'][mask]
        elif ttl_interval == 10:
            frame_number = fp_data['FrameCounter']

        ttl_diff = np.diff(daq_data.loc[daq_data['photometry_ttl'] == 1].index)
        while ttl_diff.max() > ttl_interval * 3:
            print(f'Big gaps: {ttl_diff.max()}')
            ttl_id = np.where(ttl_diff == ttl_diff.max())[0][0]
            real_id = daq_data.loc[daq_data['photometry_ttl'] == 1].index[ttl_id]
            daq_data.iloc[:int(real_id + ttl_diff.max() - ttl_interval), :] = 0
            daq_data.loc[np.where(daq_data['photometry'].diff() == 1)[0], 'photometry_ttl'] = 1
            ttl_diff = np.diff(daq_data.loc[daq_data['photometry_ttl'] == 1].index)

        photometry_ttl = np.where(daq_data['photometry_ttl'] == 1)[0].astype(int)
        assert (np.abs(photometry_ttl.size - frame_number.size) <= 15)
        fp_data.loc[frame_number, 'daq_timestamp'] = photometry_ttl[:len(frame_number)]

        daq_data.loc[np.where(daq_data['bpod'].diff() == 1)[0], 'ttl_on'] = 1
        daq_data.loc[np.where(daq_data['bpod'].diff() == -1)[0], 'ttl_off'] = 1
        daq_data.loc[np.where(daq_data['bpod'].diff() == 1)[0], 'ttl_duration'] = \
            daq_data.loc[daq_data['ttl_off'] == 1].index - daq_data.loc[daq_data['ttl_on'] == 1].index
        # Valve and error tones have pulses > 100
        daq_data.loc[daq_data['ttl_duration'] > 105, 'feedback_times'] = 1

        n_daq_trials = (daq_data['feedback_times'] == 1).sum()
        if n_daq_trials == trials['feedback_times'].size:
            daq_data.loc[daq_data['feedback_times'] == 1, 'bpod_times'] = trials['feedback_times']
        elif n_daq_trials - trials['feedback_times'].size == 1:
            daq_data.loc[daq_data['feedback_times'] == 1, 'bpod_times'] = np.r_[trials['feedback_times'], np.nan]
        else:
            assert n_daq_trials == trials['feedback_times'].size, "Trials don't match up"

        daq_data['bpod_times'].interpolate(inplace=True)
        # Set values after last pulse to nan
        bpod_column = daq_data.columns.get_loc("bpod_times")
        daq_data.iloc[np.where(daq_data['bpod_times'] == daq_data['bpod_times'].max())[0][1]:, bpod_column] = np.nan

        fp_data.loc[frame_number, 'bpod_times'] = daq_data['bpod_times'].values[
            fp_data.loc[frame_number, 'daq_timestamp'].values.astype(int)]

        use_times = ~fp_data['bpod_times'].isna()

        fcn = interpolate.interp1d(fp_data['Timestamp'][use_times].values, fp_data['bpod_times'][use_times].values,
                                   fill_value="extrapolate")

        ts = fcn(fp_data['Timestamp'].values)

        return ts


class FibrePhotometryPreprocess(PhotometryPreprocess):

    def _run(self, **kwargs):
        _, out_files = FibrePhotometry(self.session_path, collection=self.device_collection).extract(
            regions=self.regions, path_out=self.session_path.joinpath('alf', 'photometry'), save=True)
        return out_files
