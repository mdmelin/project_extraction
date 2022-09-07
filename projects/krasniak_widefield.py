from ibllib.io.extractors.widefield import Widefield as BaseWidefield
from pathlib import Path
from ibllib.io.extractors.ephys_fpga import get_main_probe_sync, get_sync_fronts
import numpy as np
from labcams.io import parse_cam_log
import ibllib.exceptions as err
from ibllib.io.video import get_video_meta
from ibllib.pipes.base_tasks import WidefieldTask
import neurodsp as dsp
import logging
import wfield.cli as wfield_cli

_logger = logging.getLogger('ibllib')

DEFAULT_WIRING_MAP = {
    3: 470,
    2: 405
}


class Widefield(BaseWidefield):

    def preprocess(self, fs=30, functional_channel=1, nbaseline_frames=30, k=200, nchannels=2):

        # No motion correction for Krasniak

        # COMPUTE AVERAGE FOR BASELINE
        wfield_cli._baseline(str(self.data_path), nbaseline_frames, nchannels=nchannels)
        # DATA REDUCTION
        wfield_cli._decompose(str(self.data_path), k=k, nchannels=nchannels)
        # HAEMODYNAMIC CORRECTION
        # check if it is 2 channel
        dat = wfield_cli.load_stack(str(self.data_path), nchannels=nchannels)
        if dat.shape[1] == 2:
            del dat
            wfield_cli._hemocorrect(str(self.data_path), fs=fs, functional_channel=functional_channel)

    def sync_timestamps(self, save=False, save_paths=None, **kwargs):

        if save and save_paths:
            assert len(save_paths) == 3, 'Must provide save_path as list with 3 paths'
            for save_path in save_paths:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        filepath = next(self.data_path.glob('*.camlog'))
        fpga_sync, chmap = get_main_probe_sync(self.session_path)
        bpod = get_sync_fronts(fpga_sync, chmap['bpod'])
        logdata, led, sync, ncomm = parse_cam_log(filepath, readTeensy=True)
        if bpod.times.size == 0:
            raise err.SyncBpodFpgaException('No Bpod event found in FPGA. No behaviour extraction. '
                                            'Check channel maps.')

        # Check that the no. of syncs from bpod and teensy match
        assert len(bpod['times']) == len(sync), 'Number of detected sync pulses on bpod and teensy do not match'

        # convert to seconds
        fcn, drift, iteensy, ifpga = dsp.utils.sync_timestamps(sync.timestamp.values / 1e3, bpod['times'], return_indices=True)

        _logger.debug(f'Widefield-FPGA clock drift: {drift} ppm')
        assert led.frame.is_monotonic_increasing
        video_path = next(self.data_path.glob('imaging.frames.mov'))
        video_meta = get_video_meta(video_path)

        diff = len(led) - video_meta.length
        if diff < 0:
            raise ValueError('More frames than timestamps detected')
        if diff > 2:
            raise ValueError('Timestamps and frames differ by more than 2')

        led = led[0:video_meta.length]

        # Find led times that are outside of the sync pulses
        led_times = np.copy(led.timestamp.values)
        pre_times = led_times < np.min(sync.timestamp)
        post_times = led_times > np.max(sync.timestamp)
        led_times[pre_times] = np.nan
        led_times[post_times] = np.nan

        # Interpolate frames that lie within sync pulses timeframe
        widefield_times = fcn(led_times / 1e3)
        kp = ~np.isnan(widefield_times)
        # Extrapolate times that lie outside sync pulses timeframe (i.e before or after)
        pol = np.polyfit(led_times[kp] / 1e3, widefield_times[kp], 1)
        extrap_vals = np.polyval(pol, led.timestamp.values / 1e3)
        widefield_times[~kp] = extrap_vals[~kp]

        assert np.all(np.diff(widefield_times) > 0)

        # Now extract the LED channels and meta data
        # Load channel meta and wiring map
        channel_meta_map = self._channel_meta(kwargs.get('light_source_map'))
        channel_wiring = self._channel_wiring()
        channel_id = np.empty_like(led.led.values)

        for _, d in channel_wiring.iterrows():
            mask = led.led.values == d['LED']
            if np.sum(mask) == 0:
                raise err.WidefieldWiringException
            channel_id[mask] = channel_meta_map.get(channel_meta_map['wavelength'] == d['wavelength']).index[0]

        if save:
            save_time = save_paths[0] if save_paths else self.data_path.joinpath('timestamps.npy')
            save_led = save_paths[1] if save_paths else self.data_path.joinpath('led.npy')
            save_meta = save_paths[2] if save_paths else self.data_path.joinpath('led_properties.csv')
            save_paths = [save_time, save_led, save_meta]
            np.save(save_time, widefield_times)
            np.save(save_led, channel_id)
            channel_meta_map.to_csv(save_meta)

            return save_paths
        else:
            return widefield_times, channel_id, channel_meta_map


class WidefieldSyncKrasniak(WidefieldTask):
    priority = 60
    level = 1
    force = False
    job_size = 'small'
    signature = {
        'input_files': [('imaging.frames.mov', 'raw_widefield_data', True),
                        ('widefieldEvents.raw.camlog', 'raw_widefield_data', True),
                        ('_spikeglx_sync.channels.npy', 'raw_ephys_data', True),
                        ('_spikeglx_sync.polarities.npy', 'raw_ephys_data', True),
                        ('_spikeglx_sync.times.npy', 'raw_ephys_data', True)],
        'output_files': [('imaging.times.npy', 'alf/widefield', True),
                         ('imaging.imagingLightSource.npy', 'alf/widefield', True),
                         ('imagingLightSource.properties.htsv', 'alf/widefield', True)]
    }

    def _run(self):

        self.wf = Widefield(self.session_path)
        save_paths = [self.session_path.joinpath(sig[1], sig[0]) for sig in self.signature['output_files']]
        out_files = self.wf.sync_timestamps(save=True, save_paths=save_paths)

        # TODO QC

        return out_files


class WidefieldPreprocessKrasniak(WidefieldTask):

    priority = 80
    job_size = 'large'

    @property
    def signature(self):
        signature = {
            'input_files': [('*.dat', self.device_collection, True),
                            ('widefieldEvents.raw.*', self.device_collection, True)],
            'output_files': [('widefieldChannels.frameAverage.npy', 'alf/widefield', True),
                             ('widefieldU.images.npy', 'alf/widefield', True),
                             ('widefieldSVT.uncorrected.npy', 'alf/widefield', True),
                             ('widefieldSVT.haemoCorrected.npy', 'alf/widefield', True)]
        }
        return signature

    def _run(self, **kwargs):
        self.wf = Widefield(self.session_path)
        _, out_files = self.wf.extract(save=True, extract_timestamps=False)
        return out_files

    def tearDown(self):
        super(WidefieldPreprocessKrasniak, self).tearDown()
        self.wf.remove_files()
