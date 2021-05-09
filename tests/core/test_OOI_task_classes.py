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
import unittest

import sys
sys.path.append("D:/Code/eotopia/core")
from data_OOI_utilities import DataParser
from OOI_task_classes import OOITask

logging.basicConfig(level=logging.DEBUG)

class TestException(BaseException):
    def __init__(self, param1, param2):
        # accept two parameters as opposed to BaseException, which just accepts one
        super().__init__()
        self.param1 = param1
        self.param2 = param2



class ExceptionTestingTask(OOITask):
    def __init__(self, task_arg):
        self.task_arg = task_arg

    def execute(self, exec_param):
        # try raising a subclassed exception with an unsupported 
        # __init__ arguments signature
        if self.task_arg == 'test_exception':
            raise TestException(1, 2)

        # try raising a subclassed exception with an unsupported 
        # __init__ arguments signature without initializing it
        if self.task_arg == 'test_exception_fail':
            raise TestException

        # raise one of the standard errors
        if self.task_arg == 'value_error':
            raise ValueError('Testing value error.')

        return self.task_arg + ' ' + exec_param

class TestOOITask(unittest.TestCase):
    class PlusOneTask(OOITask):

        @staticmethod
        def execute(x):
            return x + 1

    def test_call_equals_transform(self):
        t = self.PlusOneTask()
        self.assertEqual(t(1), t.execute(1), 
                         msg="t(x) should given the same result as t.execute(x)")

class TestCompositeTask(unittest.TestCase):
    class MultTask(OOITask):

        def __init__(self, num):
            self.num = num

        def execute(self, x):
            return (x + 1) * self.num

    def test_chained(self):
        composite = self.MultTask(1) * self.MultTask(2) * self.MultTask(3)

        for i in range(5):
            self.assertEqual(composite(i), 6 * i + 9)

    def test_execution_handling(self):
        task = ExceptionTestingTask('test_exception')
        self.assertRaises(TestException, task, 'test')

        task = ExceptionTestingTask('success')
        self.assertEqual(task('test'), 'success test')

        for parameter, exception_type in\
            [('test_exception_fail', TypeError), ('value_error', ValueError)]:
            task = ExceptionTestingTask(parameter)
            self.assertRaises(exception_type, task, 'test')
            try:
                task('test')
            except exception_type as exception:
                message = str(exception)
                self.assertTrue(message.startswith('During execution of\
                                                   task ExceptionTestingTask: '))

if __name__ == '__main__':
    unittest.main()

