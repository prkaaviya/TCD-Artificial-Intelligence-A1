"""
Maze solving algorithms package.
"""

from solvers.uninformed import DFSSolver, BFSSolver
from solvers.informed import AStarSolver
from solvers.mdp import MDPValueIterationSolver, MDPPolicyIterationSolver

def get_solver_class(algorithm):
    """
    Return the appropriate solver class based on the algorithm name.
    """
    solvers = {
        "DFS": DFSSolver,
        "BFS": BFSSolver,
        "A*": AStarSolver,
        "MDP_VALUE": MDPValueIterationSolver,
        "MDP_POLICY": MDPPolicyIterationSolver
    }

    if algorithm not in solvers:
        raise ValueError(f"Unknown algorithm: {algorithm}. \
            Available algorithms: {list(solvers.keys())}")

    return solvers[algorithm]
