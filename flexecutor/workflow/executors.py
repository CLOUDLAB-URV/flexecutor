from abc import abstractmethod, ABC

from flexecutor.workflow.task import Task
from flexecutor.workflow.taskfuture import TaskFuture


class Executor(ABC):
    """
    Abstract base class for executors
    """

    def __init__(self):
        pass

    @abstractmethod
    def execute(
            self,
            task: Task,
            *args,
            **kwargs
    ) -> TaskFuture:
        """
        Execute a task and wait for it to finish

        :param task: Task to execute
        :return: Output data of the tasks
        """
        pass


class CallableExecutor(Executor):
    """
    Executor that executes a callable
    """

    def __init__(self):
        super().__init__()

    def execute(
            self,
            task: Task,
            *args,
            **kwargs
    ) -> TaskFuture:
        """
        Execute a task and wait for it to finish

        :param task: Task to execute
        :return: Output data of the tasks
        """
        future = task(*args, **kwargs)
        task.executor.wait(future)
        return TaskFuture(future)
