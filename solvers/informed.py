"""
informed.py - Maze solver for informed search algorithms: A*.
"""

import heapq
import numpy as np
from solvers.base import MazeSolverBase

class AStarSolver(MazeSolverBase):
    """
    Maze solver using A* Search.
    """
