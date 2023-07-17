import logging
import shutil
import unittest.mock

import numpy as np
from iblutil.util import Bunch
from projects.krasniak_widefield import WidefieldSyncKrasniak

from ci.tests import base

_logger = logging.getLogger('ibllib')


class TestWidefieldSync(base.IntegrationTest):
    patch = None  # A mock of get_video_meta
    video_meta = Bunch()

    def setUp(self):
        self.session_path = self.default_data_root().joinpath(
            'widefield', 'widefieldChoiceWorld', 'CSK-im-011', '2021-07-29', '001')
        if not self.session_path.exists():
            return
        self.alf_folder = self.session_path.joinpath('alf', 'widefield')

        self.video_meta.length = 183340
        self.patch = unittest.mock.patch('projects.krasniak_widefield.get_video_length',
                                         return_value=self.video_meta['length'])
        self.patch.start()

    def test_sync(self):

        wf = WidefieldSyncKrasniak(self.session_path)
        status = wf.run()
        assert status == 0

        for exp_files in wf.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            assert file.exists()
            assert file in wf.outputs

        # Check integrity of outputs
        times = np.load(self.alf_folder.joinpath('imaging.times.npy'))
        assert len(times) == self.video_meta['length']
        assert np.all(np.diff(times) > 0)
        leds = np.load(self.alf_folder.joinpath('imaging.imagingLightSource.npy'))
        assert leds[0] == 1
        assert np.array_equal(np.unique(leds), np.array([1, 2]))

    def test_wrong_wiring(self):
        wiring_file = self.session_path.joinpath(
            'raw_widefield_data', 'widefieldChannels.wiring.htsv')
        bk_wiring_file = wiring_file.parent.joinpath(wiring_file.name + '.bk')
        wiring_file.rename(bk_wiring_file)
        self.addCleanup(bk_wiring_file.rename, wiring_file)

        wf = WidefieldSyncKrasniak(self.session_path)
        status = wf.run()
        assert status == -1
        assert "WidefieldWiringException" in wf.log

    def tearDown(self):
        if self.alf_folder.exists():
            shutil.rmtree(self.alf_folder.parent)
        self.patch.stop()
