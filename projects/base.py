import logging
import importlib

_logger = logging.getLogger('ibllib')


def get_pipeline(task_type):
    """
    Get the pipeline Task from task type - returns None if the task is not found
    :param task_type: string that should match the module name
    :return:
    """
    try:
        mdl = importlib.import_module(f"projects.{task_type}")
    except:
        ModuleNotFoundError()
        _logger.error({f"Import error: projects.{task_type} not found: does this project exists ?"})
        return
    return mdl.__pipeline__
