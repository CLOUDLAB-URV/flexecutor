from flexexecutor.core.stage import WorkflowStage
from flexexecutor.core.optimization import OptimizationProblemSolver


class Scheduler:
    def __init__(
        self,
        workflow_stage: WorkflowStage,
        memory_configs=[1792, 2560, 3584, 5120, 7168, 10240],
        vcpu_configs=[0.6, 1, 1.5, 2, 2.5, 3, 4],
        parallel_configs=[1, 4, 6, 8, 16, 32],
    ):
        self.num_funcs = []
        self.num_vcpus = []
        self.mem_configs = []
        self.bound_type = None
        self.bound = None
        self.workflow_stage = workflow_stage
        self.optimization_problem_solver = OptimizationProblemSolver()

    def search_config(self):
        """
        objective_func = None
        constraint_func = None
        self.obj_params = self.worklof_stage.obj_params

        self.solver(
            self.workflowstage,
            objective_func,
            constraint_func,
            self.bound,
            self.obj_params,
            self.cons_params,
        )
        res = self.solver.iter_solve(init_vals, x_bound)
        """

        print("num_funcs:", 20)
        print("num_vcpus:", 10)
        print("mem:", 1024)
        print("chunk_size:", 1024)

    def predict(self):
        pass
