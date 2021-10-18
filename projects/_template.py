from collections import OrderedDict

from ibllib.pipes import tasks


class TemplateTask(tasks.Task):
    cpu = 1
    io_charge = 90
    level = 1
    signature = [
        ('_ibl_trials.laserIntervals.npy', 'alf', True),
        ('_ibl_trials.laserProbability.npy', 'alf', True),
        ('_ibl_trials.intervals.npy', 'alf', True)]

    def _run(self):
        pass


class TemplatePipeline(tasks.Pipeline):
    label = __name__

    def __init__(self, session_path=None, **kwargs):
        super(TemplateTask, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks["EphysRegisterRaw"] = TemplateTask(self.session_path)
        self.tasks = tasks


__pipeline__ = TemplatePipeline
