# -*- coding: utf-8 -*-
"""
The ooiworkflow module, together with ooitask and ooidata, provides core 
building blocks for specifying and executing workflows.

A workflow is a directed (acyclic) graph composed of instances of OOITask objects. 
Each task may take as input the results of other tasks and external arguments. 
The external arguments are passed anew each time the workflow is executed. 
The workflow builds the computational graph, performs dependency resolution, 
and executes the tasks.
If the input graph is cyclic, the workflow raises a `CyclicDependencyError`.
The result of a workflow execution is an immutable mapping from tasks to results. 
The result contains tasks with zero out-degree (i.e. terminal tasks).

@author: freeridingeo
"""

import collections
import logging
import warnings
import uuid
import copy

import attr

import sys
sys.path.append("D:/Code/eotopia/core")
from OOI_task_classes import OOITask

LOGGER = logging.getLogger(__name__)

class CyclicDependencyError(ValueError):
    """ 
    This error is raised when trying to initialize `OOIWorkflow` with a 
    cyclic dependency graph
    """

class OOIWorkflow:
    """ 
    A basic object for building workflows from a list of task dependencies

    Example:

            workflow = OOIWorkflow([  # task1, task2, task3 are initialized OOITasks
                (task1, [], 'My first task'),
                (task2, []),
                (task3, [task1, task2], 'Task that depends on previous 2 tasks')
            ])
    """

    def __init__(self, dependencies, task_names=None):
        """
        :param dependencies: A list of dependencies between tasks, 
            specifying the computational graph.
        :type dependencies: list(tuple or Dependency)
        """
        ## TODO
#        self.id_gen = _UniqueIdGenerator()

        if task_names:
            warnings.warn("Parameter 'task_names' could be removed.\
                          Everything can be specified with "
                          "'dependencies' parameter, including task names", 
                          DeprecationWarning, stacklevel=2)
        
        ## TODO
#        self.dependencies = self._parse_dependencies(dependencies, task_names)
        ## TODO
#        self.uuid_dict = self._set_task_uuid(self.dependencies)
        ## TODO
#        self.dag = self.create_dag(self.dependencies)
        ## TODO
#        self.ordered_dependencies = self._schedule_dependencies(self.dag)

    @staticmethod
    def _parse_dependencies(dependencies, task_names):
        """ 
        Parses dependencies and adds names of task_names

        :param dependencies: Input of dependency parameter
        :type dependencies: list(tuple or Dependency)
        :param task_names: Human-readable names of tasks
        :type task_names: dict(OOITask: str) or None
        :return: List of dependencies
        :rtype: list(Dependency)
        """
        ## TODO!
#        parsed_dependencies = [dep if isinstance(dep, Dependency) else Dependency(*dep) for dep in dependencies]
#        for dep in parsed_dependencies:
#            if task_names and dep.task in task_names:
#                dep.set_name(task_names[dep.task])
#        return parsed_dependencies

    def _set_task_uuid(self, dependencies):
        """ 
        Adds universally unique user ids (UUID) to each task of the workflow
        
        :param dependencies: The list of dependencies between tasks 
            defining the computational graph
        :type dependencies: list(Dependency)
        :return: A dictionary mapping UUID to dependencies
        :rtype: dict(str: Dependency)
        """
        uuid_dict = {}
        for dep in dependencies:
            task = dep.task
            ## TODO!
            # if task.private_task_config.uuid in uuid_dict:
            #     raise ValueError('EOWorkflow cannot execute the same instance of EOTask multiple times')

            # task.private_task_config.uuid = self.id_gen.get_next()
            # uuid_dict[task.private_task_config.uuid] = dep

        return uuid_dict

    def create_dag(self, dependencies):
        """ 
        Creates a directed graph from dependencies
        
        :param dependencies: A list of Dependency objects
        :type dependencies: list(Dependency)
        :return: A directed graph of the workflow
        :rtype: DirectedGraph
        """
        ## TODO!
        # dag = DirectedGraph()
        # for dep in dependencies:
        #     for vertex in dep.inputs:
        #         task_uuid = vertex.private_task_config.uuid
        #         if task_uuid not in self.uuid_dict:
        #             raise ValueError('Task {}, which is an input of a task {}, is not part of the defined '
        #                              'workflow'.format(vertex.__class__.__name__, dep.name))
        #         dag.add_edge(self.uuid_dict[task_uuid], dep)
        #     if not dep.inputs:
        #         dag.add_vertex(dep)
        # return dag

    @staticmethod
    def _schedule_dependencies(dag):
        """ 
        Computes an ordering < of tasks so that for any two tasks t and t' 
        we have that if t depends on t' then t' < t. 
        In words, all dependencies of a task precede the task in this ordering.
        
        :param dag: A directed acyclic graph representing dependencies between tasks.
        :type dag: DirectedGraph
        :return: A list of topologically ordered dependecies
        :rtype: list(Dependency)
        """

