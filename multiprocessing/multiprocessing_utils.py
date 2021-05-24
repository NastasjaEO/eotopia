# -*- coding: utf-8 -*-
"""
Created on Mon May 24 11:11:47 2021

@author: freeridingeo
"""

import os
from io import StringIO
import tempfile
import platform
import dill

import inspect
import tblib.pickling_support
import subprocess as sp
import pathos.multiprocessing as mp

import sys
sys.path.append("D:/Code/eotopia/utils")
from dict_utils import dictmerge

class HiddenPrints:
    """
    | Suppress console stdout prints, i.e. redirect them to a temporary string object.
    | Adapted from https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
    Examples
    --------
    >>> with HiddenPrints():
    >>>     print('foobar')
    >>> print('foobar')
    """
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = StringIO()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

class ExceptionWrapper(object):
    """
    | class for enabling traceback pickling in function multiprocess
    | https://stackoverflow.com/questions/6126007/python-getting-a-traceback-from-a-multiprocessing-process
    | https://stackoverflow.com/questions/34463087/valid-syntax-in-both-python-2-x-and-3-x-for-raising-exception
    """
    
    def __init__(self, ee):
        self.ee = ee
        __, __, self.tb = sys.exc_info()
    
    def re_raise(self):
        if sys.version_info[0] == 3:
            def reraise(tp, value, tb=None):
                raise tp.with_traceback(tb)
        else:
            exec("def reraise(tp, value, tb=None):\n    raise tp, value, tb\n")
        reraise(self.ee, None, self.tb)

def multicore(function, cores, multiargs, **singleargs):
    """
    wrapper for multicore process execution
    Parameters
    ----------
    function
        individual function to be applied to each process item
    cores: int
        the number of subprocesses started/CPUs used;
        this value is reduced in case the number of subprocesses is smaller
    multiargs: dict
        a dictionary containing sub-function argument names as keys and lists 
        of arguments to be distributed among the processes as values
    singleargs
        all remaining arguments which are invariant among the subprocesses
    Returns
    -------
    None or list
        the return of the function for all subprocesses
    Notes
    -----
    - all `multiargs` value lists must be of same length, i.e. all argument 
        keys must be explicitly defined for each subprocess
    - all function arguments passed via `singleargs` must be provided with the 
        full argument name and its value (i.e. argname=argval); 
        default function args are not accepted
    - if the processes return anything else than None, this function will 
        return a list of results
    - if all processes return None, this function will be of type void
    Examples
    --------
    >>> def add(x, y, z):
    >>>     return x + y + z
    >>> multicore(add, cores=2, multiargs={'x': [1, 2]}, y=5, z=9)
    [15, 16]
    >>> multicore(add, cores=2, multiargs={'x': [1, 2], 'y': [5, 6]}, z=9)
    [15, 17]
    """
    tblib.pickling_support.install()
    check = inspect.getfullargspec(function)
    varkw = check.varkw

    if not check.varargs and not varkw:
        multiargs_check = [x for x in multiargs if x not in check.args]
        singleargs_check = [x for x in singleargs if x not in check.args]
        if len(multiargs_check) > 0:
            raise AttributeError('incompatible multi arguments: {0}'\
                                 .format(', '.join(multiargs_check)))
        if len(singleargs_check) > 0:
            raise AttributeError('incompatible single arguments: {0}'\
                                 .format(', '.join(singleargs_check)))
    
    # compare the list lengths of the multi arguments and raise errors 
    # if they are of different length
    arglengths = list(set([len(multiargs[x]) for x in multiargs]))
    if len(arglengths) > 1:
        raise AttributeError('multi argument lists of different length')

    # prevent starting more threads than necessary
    cores = cores if arglengths[0] >= cores else arglengths[0]
    
    # create a list of dictionaries each containing the arguments for individual
    # function calls to be passed to the multicore processes
    processlist = [dictmerge(dict([(arg, multiargs[arg][i])\
                                   for arg in multiargs]), singleargs)
                   for i in range(len(multiargs[list(multiargs.keys())[0]]))]

    if platform.system() == 'Windows':
        
        # in Windows parallel processing needs to strictly be in a "if __name__ == '__main__':" wrapper
        # it was thus necessary to outsource this to a different script and try to serialize all input for sharing objects
        # https://stackoverflow.com/questions/38236211/why-multiprocessing-process-behave-differently-on-windows-and-linux-for-global-o
        
        # a helper script to perform the parallel processing
        script = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                              'multicore_helper.py')
        
        # a temporary file to write the serialized function variables
        tmpfile = os.path.join(tempfile.gettempdir(), 'spatialist_dump')

        # check if everything can be serialized
        if not dill.pickles([function, cores, processlist]):
            raise RuntimeError('cannot fully serialize function arguments;\n'
                               ' see https://github.com/uqfoundation/dill for supported types')
        
        # write the serialized variables
        with open(tmpfile, 'wb') as tmp:
            dill.dump([function, cores, processlist], tmp, byref=False)
        
        # run the helper script
        proc = sp.Popen([sys.executable, script], stdin=sp.PIPE, stderr=sp.PIPE)
        out, err = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(err.decode())
        
        # retrieve the serialized output of the processing which was written to the temporary file by the helper script
        with open(tmpfile, 'rb') as tmp:
            result = dill.load(tmp)
        return result
    else:
        results = None
        
        def wrapper(**kwargs):
            try:
                return function(**kwargs)
            except Exception as e:
                return ExceptionWrapper(e)
        
        # block printing of the executed function
        with HiddenPrints():
            # start pool of processes and do the work
            try:
                pool = mp.Pool(processes=cores)
            except NameError:
                raise ImportError("package 'pathos' could not be imported")
            results = pool.imap(lambda x: wrapper(**x), processlist)
            pool.close()
            pool.join()
        
        i = 0
        out = []
        for item in results:
            if isinstance(item, ExceptionWrapper):
                item.ee = type(item.ee)(str(item.ee) +
                                        "\n(called function '{}' with args {})"
                                        .format(function.__name__, processlist[i]))
                raise (item.re_raise())
            out.append(item)
            i += 1
        
        # evaluate the return of the processing function;
        # if any value is not None then the whole list of results is returned
        eval = [x for x in out if x is not None]
        if len(eval) == 0:
            return None
        else:
            return out
