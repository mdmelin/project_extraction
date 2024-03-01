"""Template boilerplate code for custom task protocol data extraction.

For an example of how to modify Bpod trials extraction (with either Bpod only or unchanged DAQ time alignment) check out
'projects.neuromodulators'.  For an example of custom task QC, see 'projects.samuel_cuedBiasedChoiceWorld'.
"""
from collections import OrderedDict

from ibllib.pipes.tasks import Pipeline
from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsBpod
import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes


class TemplateBpodTrialsExtractor(BaseBpodTrialsExtractor):
    """
    Extract ALF trials files for a custom Bpod protocol.

    Attributes
    ----------
    bpod_trials : list of dict
        The Bpod trials loaded from the _iblrig_taskData.raw.jsonable file.
    settings : dict
        The Bpod settings loaded from the _iblrig_taskSettings.raw.json file.
    session_path : pathlib.Path
        The absolute session path.
    task_collection : str
        The raw task data collection

    """
    var_names = ('laser_intervals', 'laser_probability', 'intervals')
    """tuple of str: The variable names of each extracted dataset. The dict returned by `_extract`
    must use the values in `var_names` as its keys.
    """

    save_names = ('_ibl_trials.laserIntervals.npy', None, '_ibl_trials.intervals.npy')
    """tuple of str: The dataset filenames of the var_names returned by `_extract`. The tuple length
    must match `var_names`. None values are not automatically saved to file. This may be useful for
    extracting data you don't with to save, but will use for QC or debugging.
    """

    def _extract(self, extractor_classes=None, **kwargs) -> dict:
        """
        Extract the Bpod trial events.

        Saving of the datasets to file is handled by the superclass.

        Returns
        -------
        dict
            A dictionary of Bpod trial events. The keys are defined in the `var_names` attribute.

        See Also
        --------
        ibllib.io.extractors.habituation_trials.HabituationTrials - A good example of how trials
            data can be extracted from raw data.

        Examples
        --------
        Get all detected TTLs. These should be stored for QC purposes

        >>> self.frame2ttl, self.audio = raw.load_bpod_fronts(self.session_path, data=self.bpod_trials)

        These are the frame2TTL pulses as a list of lists, one per trial

        >>> ttls = [raw.get_port_events(tr, 'BNC1') for tr in self.bpod_trials]

        Extract datasets common to your adapted protocol, e.g. contrast, stim on, feedback, etc.

        >>> from ibllib.io.extractors.biased_trials import ContrastLR
        >>> from ibllib.io.extractors.training_trials import FeedbackTimes, StimOnTriggerTimes, GoCueTimes
        >>> training = [ContrastLR, FeedbackTimes, GoCueTimes, StimOnTriggerTimes]
        >>> out, _ = run_extractor_classes(
        ...     training, session_path=self.session_path, save=False,  bpod_trials=self.bpod_trials,
        ...     settings=self.settings, task_collection=self.task_collection)
        """
        ...


class TemplateTask(ChoiceWorldTrialsBpod):
    """A template behaviour task.

    If the task protocol is a simple Bpod-only task (without an extra DAQ), you do not need a
    separate behaviour task. Instead, create a new ibllib.io.extractors.BaseBpodTrialsExtractor
    subclass (see above). You may need to create a custom Task if you want to run your own QC,
    however for this you can simply overload the `run_qc` method with your preferred QC class in
    the kwargs (see `projects.samuel_cuedBiasedChoiceWorld` for example).
    """
    cpu = 1
    io_charge = 90
    level = 1
    signature = [
        ('_ibl_trials.laserIntervals.npy', 'alf', True),
        ('_ibl_trials.laserProbability.npy', 'alf', True),
        ('_ibl_trials.intervals.npy', 'alf', True)]

    def extract_behavior(self, **kwargs):
        """Extract the Bpod trials data.

        Parameters
        ----------
        kwargs

        Returns
        -------
        dict
            The extracted trials datasets.
        list of pathlib.Path
            The saved dataset filepaths.
        """
        # First determine the extractor from the task protocol
        bpod_trials, out_files = ChoiceWorldTrialsBpod._extract_behaviour(self, save=False, **kwargs)

        ...  # further trials manipulation, etc.
        """
        Sync Bpod trials to DAQ, etc. For syncing trials data to a DAQ, see
        ibllib.io.extractors.ephys_fpga.TrialsFpga and ibllib.pipes.behavior_tasks.ChoiceWorldTrialsNidq.
        If using such a trials extractor, run it here and assign the object to `self.extractor`.
        This can be accessed by `run_qc` to access the loaded raw IO data.
        """
        dsets = bpod_trials

        return dsets, out_files

    def run_qc(self, trials_data=None, update=True):
        """
        Run task QC.

        Parameters
        ----------
        trials_data : dict
            The extracted trials datasets.
        update : bool
            If True, updates Alyx with the QC outcomes.

        Returns
        -------
        ibllib.qc.base.QC
            The QC object.
        """
        if not self.extractor or trials_data is None:
            trials_data, _ = self.extract_behaviour(save=False)
        if not trials_data:
            raise ValueError('No trials data found')

        # Compile task data for QC
        qc_extractor = TaskQCExtractor(self.session_path, lazy=True, sync_collection=self.sync_collection, one=self.one,
                                       sync_type=self.sync, task_collection=self.collection)
        qc_extractor.data = qc_extractor.rename_data(trials_data)
        if type(self.extractor).__name__ == 'HabituationTrials':
            qc = HabituationQC(self.session_path, one=self.one, log=_logger)
        else:
            qc = TaskQC(self.session_path, one=self.one, log=_logger)
            qc_extractor.wheel_encoding = 'X1'
        qc_extractor.settings = self.extractor.settings
        qc_extractor.frame_ttls, qc_extractor.audio_ttls = load_bpod_fronts(
            self.session_path, task_collection=self.collection)
        qc.extractor = qc_extractor

        # Aggregate and update Alyx QC fields
        namespace = 'task' if self.protocol_number is None else f'task_{self.protocol_number:02}'
        qc.run(update=update, namespace=namespace)
        return qc


class TemplatePipeline(Pipeline):
    """(DEPRECATED) An optional legacy task pipeline."""
    label = __name__

    def __init__(self, session_path=None, **kwargs):
        super(TemplateTask, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks["EphysRegisterRaw"] = TemplateTask(self.session_path)
        self.tasks = tasks


__pipeline__ = TemplatePipeline
