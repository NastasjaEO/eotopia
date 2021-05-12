# -*- coding: utf-8 -*-
"""
This module implements the core class hierarchy for implementing tasks on OOIs.

OOI task classes are supposed to be lightweight (i.e. not too complicated), 
short, and do one thing well. 
For example, an OOI task might take as input one OOI containing cloud mask and 
return as a result the cloud coverage for that mask.

@author: freeridingeo
"""

import logging
import inspect
from abc import ABC, abstractmethod
import attr

from collections import OrderedDict
import datetime

import sys
sys.path.append("D:/Code/eotopia/core")
from data_OOI_utilities import DataParser

LOGGER = logging.getLogger(__name__)

class OOITask(ABC):

    """Base class for OOITask."""
    def __new__(cls, *args, **kwargs):
        """
        Stores initialization parameters and the order to the instance 
        attribute `init_args`.
        """
        self = super().__new__(cls)

        init_args = OrderedDict()
        for arg, value in\
            zip(inspect.getfullargspec(self.__init__).\
                args[1: len(args) + 1], args):
            init_args[arg] = repr(value)
        for arg in\
            inspect.getfullargspec(self.__init__).args[len(args) + 1:]:
            if arg in kwargs:
                init_args[arg] = repr(kwargs[arg])

        self.private_task_config = _PrivateTaskConfig(init_args=init_args)
        return self

    def __mul__(self, other):
        """Creates a composite task of this and passed task."""
        return CompositeTask(other, self)

    def __call__(self, *oois, monitor=False, **kwargs):
        """Executes the task."""
        if monitor:
            return self.execute_and_monitor(*oois, **kwargs)
        return self._execute_handling(*oois, **kwargs)

    def execute_and_monitor(self, *oois, **kwargs):
        """ 
        In the current version nothing additional happens in this method
        """
        return self._execute_handling(*oois, **kwargs)

    def _execute_handling(self, *oois, **kwargs):
        """ 
        Handles measuring execution time and error propagation
        """
        self.private_task_config.start_time = datetime.datetime.now()

        try:
            return_value = self.execute(*oois, **kwargs)
            self.private_task_config.end_time = datetime.datetime.now()
            return return_value
        except BaseException as exception:
            traceback = sys.exc_info()[2]

            # Some special exceptions don't accept an error message 
            # as a parameter and raise a TypeError in such case.
            try:
                errmsg = 'During execution of task {}: {}'.\
                    format(self.__class__.__name__, exception)
                extended_exception = type(exception)(errmsg)
            except TypeError:
                extended_exception = exception
            raise extended_exception.with_traceback(traceback)

    @abstractmethod
    def execute(self, *oois, **kwargs):
        """ Implement execute function
        """
        raise NotImplementedError

    @staticmethod
    def _parse_data(data, new_names=False, rename_function=None, 
                    default_data_type=None, allowed_data_types=None):
        return DataParser(data, new_names=new_names, rename_function=rename_function,
                             default_data_type=default_data_type, 
                             allowed_data_types=allowed_data_types)

@attr.s(eq=False)
class _PrivateTaskConfig:
    """ 
    A container for general OOITask parameters required during 
    OOIWorkflow and OOIExecution.
    
    :param init_args: A dictionary of parameters and values used for 
        OOITask initialization
    :type init_args: OrderedDict
    :param uuid: An unique hexadecimal identifier string a task gets 
        in OOIWorkflow
    :type uuid: str or None
    :param start_time: Time when task execution started
    :type start_time: datetime.datetime or None
    :param end_time: Time when task execution ended
    :type end_time: datetime.datetime or None
    """
    init_args = attr.ib()
    uuid = attr.ib(default=None)
    start_time = attr.ib(default=None)
    end_time = attr.ib(default=None)

    def __add__(self, other):
        return _PrivateTaskConfig(init_args=OrderedDict(list(self.init_args.items())\
                                                + list(other.init_args.items())))

class CompositeTask(OOITask):
    """
    Creates a task that is composite of two tasks.
    
    Note: Instead of directly using this task it might be more convenient 
    to use `'*'` operation between tasks.
    Example: `composite_task = task1 * task2`
    
    :param ooitask1: Task which will be executed first
    :type ooitask1: OOITask
    :param ooitask2: Task which will be executed on results of first task
    :type ooitask2: OOITask
    """
    def __init__(self, ooitask1, ooitask2):
        self.ooitask1 = ooitask1
        self.ooitask2 = ooitask2

        self.private_task_config =\
            ooitask1.private_task_config + ooitask2.private_task_config

    def execute(self, *oois, **kwargs):
        return self.ooitask2.execute(self.ooitask1.execute(*oois, **kwargs))
