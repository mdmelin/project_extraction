"""Integration test for _sp_passiveVideo extraction."""

import unittest
import ibllib.pipes.dynamic_pipeline as dyn
from one.api import ONE


@unittest.skip('Integration test')
class TestPassiveVideo(unittest.TestCase):
    """Test _sp_passiveVideo extraction.

    The test session is a little short. The second 30Hz video repeat does not
    sync very well due to so few pulses. The first 60Hz video repeat is fine.
    """
    required_files = ['test/2025-01-27/002', 'SP061/2025-01-27/001']

    def test_extraction(self):
        one = ONE()
        session_path = one.cache_dir.joinpath(
            'cortexlab', 'Subjects', self.required_files[1])
        tasks = dyn._get_trials_tasks(session_path)
        tasks = [t for t in tasks.values() if 'passivevideo' in t.name.casefold()]
        self.assertTrue(any(tasks))
        # Test
        # task = tasks[0]  # 60Hz
        # task = tasks[1]  # 30Hz
        # SP061
        task = tasks[0]  # 60Hz
        ret = task.run(save=True, frame_rate=None)
        self.assertEqual(0, ret)
        self.assertEqual(1, len(task.outputs))
        self.assertTrue(task.outputs[0].exists())


if __name__ == '__main__':
    unittest.main()
