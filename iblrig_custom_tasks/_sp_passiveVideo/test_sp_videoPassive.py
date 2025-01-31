import unittest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
from iblrig_custom_tasks._sp_passiveVideo.task import Session, Player, MediaStats
from iblrig.test.base import TaskArgsMixin


class TestPassiveVideo(TaskArgsMixin, unittest.TestCase):

    def setUp(self):
        self.get_task_kwargs()

    def test_next_trial(self):
        self.assertRaises(NotImplementedError, Session, **self.task_kwargs)
        self.task_kwargs['hardware_settings']['MAIN_SYNC'] = False
        task = Session(log_level='DEBUG', **self.task_kwargs)
        task.video = Mock(auto_spec=Player)
        task.task_params.VIDEO = r'C:\Users\Work\Downloads\ONE\perlin-xyscale2-tscale50-comb08-5min.mp4'
        task.task_params.VIDEO = r'C:\Users\Work\Downloads\SampleVideo_1280x720_1mb.mp4'
        task.next_trial()
        task.video.play.assert_called_once_with(task.task_params.VIDEO)
        task.video.replay.assert_not_called()
        task.video.reset_mock()
        task.next_trial()
        task.video.replay.assert_called_once()

    @unittest.skip('Integration test')
    def test_save(self):
        """Test video presentation.

        This test is an integration test and requires a video file to be present in the path.
        The Bpod is not used in this test.
        """
        self.task_kwargs['hardware_settings']['MAIN_SYNC'] = False
        task = Session(log_level='DEBUG', **self.task_kwargs)
        task.task_params.VIDEO = r'C:\Users\User\Downloads\SampleVideo_1280x720_1mb.mp4'
        task.bpod = MagicMock()
        with patch.object(task, 'start_mixin_bpod'):
            task.run()
        files = list(task.paths.SESSION_FOLDER.rglob('*.*'))
        expected = [
            'transfer_me.flag', '_ibl_experiment.description_behavior.yaml',
            '_iblrig_taskSettings.raw.json', '_ibl_log.info-acquisition.log',
            '_sp_taskData.raw.pqt', '_sp_video.raw.mp4', '_sp_videoData.stats.pqt']
        self.assertCountEqual((f.name for f in files), expected)
        stats = task.video.stats
        self.assertIsInstance(stats, pd.DataFrame)
        df = pd.read_parquet(task.paths.STATS_FILE_PATH)
        self.assertTrue(all(df == stats))
        df = pd.read_parquet(task.paths.DATA_FILE_PATH)
        expected = [
            'intervals_0', 'intervals_1', 'video_runtime',
            'MediaPlayerPlaying', 'MediaPlayerEndReached']
        self.assertEqual(len(df), task.task_params.NREPEATS)
        self.assertEqual(expected, df.columns.tolist())

    def test_MediaStats(self):
        stats = MediaStats()
        fields = stats.fieldnames()
        self.assertIsInstance(fields, tuple)
        self.assertTrue(all(isinstance(f, str) for f in fields))
        field = 'demux_bitrate'
        self.assertIn(field, fields)

        values = stats.as_tuple()
        self.assertIsInstance(values, tuple)
        self.assertEqual(len(values), len(fields))


if __name__ == '__main__':
    unittest.main()
