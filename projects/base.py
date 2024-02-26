from pathlib import Path
import logging
import importlib
import warnings

from ibllib.pipes import tasks

from ibllib.io.extractors.base import get_task_extractor_type
_logger = logging.getLogger('ibllib')


def get_pipeline(task_type) -> tasks.Pipeline:
    """
    (DEPRECATED) Get the pipeline Task from task type - returns None if the task is not found.

    :param task_type: string that should match the module name
    :return:
    """
    warnings.warn('get_pipeline is deprecated. Use instructions in extraction_tasks.py instead.', DeprecationWarning)
    if isinstance(task_type, Path):
        task_type = get_task_extractor_type(task_type)
    try:
        mdl = importlib.import_module(f"projects.{task_type}")
    except ModuleNotFoundError:
        _logger.error({f"Import error: projects.{task_type} not found: does this project exist?"})
        return
    return mdl.__pipeline__
