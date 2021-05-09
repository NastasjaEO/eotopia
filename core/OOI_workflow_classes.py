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
        in_degrees = dict(dag.get_indegrees())

        independent_vertices =\
            collections.deque([vertex for vertex in dag\
                               if dag.get_indegree(vertex) == 0])
        topological_order = []
        while independent_vertices:
            v_vertex = independent_vertices.popleft()
            topological_order.append(v_vertex)

            for u_vertex in dag[v_vertex]:
                in_degrees[u_vertex] -= 1
                if in_degrees[u_vertex] == 0:
                    independent_vertices.append(u_vertex)

        if len(topological_order) != len(dag):
            raise CyclicDependencyError('Tasks do not form an acyclic graph')

        return topological_order

    def execute(self, input_args=None, monitor=False):
        """ 
        Executes the workflow
        
        :param input_args: External input arguments to the workflow. 
            They have to be in a form of a dictionary where each key is an 
            OOITask used in the workflow and each value is a dictionary or a 
            tuple of arguments.
        :type input_args: dict(OOITask: dict(str: object) or tuple(object))
        :param monitor: If True workflow execution will be monitored
        :type monitor: bool
        :return: An immutable mapping containing results of terminal tasks
        :rtype: WorkflowResults
        """
        out_degs = dict(self.dag.get_outdegrees())

        input_args = self.parse_input_args(input_args)

        ## TODO!
#        results = WorkflowResults(self._execute_tasks(input_args=input_args, 
#                   out_degs=out_degs, monitor=monitor))
#        LOGGER.debug('Workflow finished with %s', repr(results))
#        return results

    @staticmethod
    def parse_input_args(input_args):
        """ 
        Parses OOIWorkflow input arguments provided by user and raises an 
        error if something is wrong. This is done automatically in the process 
        of workflow execution
        """
        input_args = input_args if input_args else {}
        for task, args in input_args.items():
            if not isinstance(task, OOITask):
                raise ValueError('Invalid input argument {},\
                                 should be an instance of EOTask'.format(task))

            if not isinstance(args, (tuple, dict)):
                raise ValueError('Execution input arguments of each task\
                                 should be a dictionary or a tuple, for task '
                                 '{} got arguments of type {}'.\
                                     format(task.__class__.__name__, type(args)))
        return input_args

    def _execute_tasks(self, *, input_args, out_degs, monitor):
        """ 
        Executes tasks comprising the workflow in the predetermined order
        
        :param input_args: External input arguments to the workflow.
        :type input_args: Dict
        :param out_degs: Dictionary mapping vertices (task IDs) to their 
            out-degrees. (The out-degree equals the number
        of tasks that depend on this task.)
        :type out_degs: Dict
        :return: A dictionary mapping dependencies to task results
        :rtype: dict
        """
        intermediate_results = {}

        ## TODO!
        # for dep in self.ordered_dependencies:
        #     result = self._execute_task(dependency=dep,
        #                                 input_args=input_args,
        #                                 intermediate_results=intermediate_results,
        #                                 monitor=monitor)

        #     intermediate_results[dep] = result

        #     self._relax_dependencies(dependency=dep,
        #                              out_degrees=out_degs,
        #                              intermediate_results=intermediate_results)

        # return intermediate_results

    def _relax_dependencies(self, *, dependency, out_degrees, intermediate_results):
        """ 
        Relaxes dependencies incurred by ``task_id``. 
        After the task with ID ``task_id`` has been successfully executed, 
        all the tasks it depended on are updated. 
        If ``task_id`` was the last remaining dependency of a task
        ``t`` then ``t``'s result is removed from memory and, 
        depending on ``remove_intermediate``, from disk.
        
        :param dependency: A workflow dependency
        :type dependency: Dependency
        :param out_degrees: Out-degrees of tasks
        :type out_degrees: dict
        :param intermediate_results: The dictionary containing the intermediate results (needed by tasks that have yet
        to be executed) of the already-executed tasks
        :type intermediate_results: dict
        """
        ## TODO!
        # current_task = dependency.task
        # for input_task in dependency.inputs:
        #     dep = self.uuid_dict[input_task.private_task_config.uuid]
        #     out_degrees[dep] -= 1

        #     if out_degrees[dep] == 0:
        #         LOGGER.debug("Removing intermediate result for %s", 
        #                      current_task.__class__.__name__)
        #         del intermediate_results[dep]

    def get_tasks(self):
        """ 
        Returns an ordered dictionary {task_name: task} of all tasks within 
        this workflow
        
        :return: Ordered dictionary with key being task_name (str) and 
            an instance of a corresponding task from this workflow. 
            The order of tasks is the same as in which they will be executed.
        :rtype: OrderedDict
        """
        task_dict = collections.OrderedDict()
        ## TODO!
        # for dep in self.ordered_dependencies:
        #     task_name = dep.name

        #     if task_name in task_dict:
        #         count = 0
        #         while dep.get_custom_name(count) in task_dict:
        #             count += 1

        #         task_name = dep.get_custom_name(count)
        #     task_dict[task_name] = dep.task
        # return task_dict

    def get_dot(self):
        """ 
        Generates the DOT description of the underlying computational graph
        
        :return: The DOT representation of the computational graph
        :rtype: Digraph
        """
        ## TODO!
#        visualization = self._get_visualization()
#        return visualization.get_dot()

    def dependency_graph(self, filename=None):
        """ 
        Visualize the computational graph
        
        :param filename: Filename of the output image together with 
            file extension. Supported formats: `png`, `jpg`,
            `pdf`, ... . Check `graphviz` Python package for more options
        :type filename: str
        :return: The DOT representation of the computational graph, with 
            some more formatting
        :rtype: Digraph
        """
#        visualization = self._get_visualization()
#        return visualization.dependency_graph(filename=filename)

    def _get_visualization(self):
        """ 
        Helper method which provides OOIWorkflowVisualization object
        """
        ## TODO
        # pylint: disable=import-outside-toplevel,raise-missing-from
        #from eolearn.visualization import EOWorkflowVisualization
        #return EOWorkflowVisualization(self)
