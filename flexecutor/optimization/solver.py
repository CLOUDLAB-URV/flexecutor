from abc import abstractmethod, ABC


class OptimizationSolver(ABC):
    # TODO: this is just a placeholder. Different solvers can use different
    # strategies to obtain the best configuration for a DAG (such as a brute
    # force of all configs, or stochastic modelling of variables, etc.) that
    # minimize JCT or cost.
    # The specific solve method implementations will use the model within the
    # workflow stages to get predictions of latency or cost for certain
    # configurations.

    """
    This is an abstract base class for optimization solvers.
    Concrete subclasses should implement the `solve` method to solve specific optimization problems.
    """

    def __init__(self, problem):
        """
        Initialize the OptimizationSolver with a problem instance.
        """
        self.problem = problem

    @abstractmethod
    def solve(self, dag):
        """
        Solve the optimization problem.
        This is a placeholder method that needs to be implemented based on the specific problem.
        """
        raise NotImplementedError("solve method needs to be implemented in subclasses")
