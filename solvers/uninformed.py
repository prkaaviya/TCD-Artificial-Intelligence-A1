"""
uninformed.py - Maze solver for uninformed search algorithms: DFS and BFS.
"""

import numpy as np
from collections import deque
from solvers.base import *

class DFSSolver(MazeSolverBase):
    """
    Maze solver using Depth-First Search.
    """
    def __init__(self, title, maze):
        """
        Initialize the maze solver with a given maze array.

        Args:
        maze (numpy.ndarray): 2D array representing the maze 
        ('#' for walls, '.' for paths, 'S' for start, 'G' for goal).
        """
        super().__init__(title, maze)

        self.nodes_explored = 0
        self.nodes_available = len(list(zip(*np.where(maze == '.'))))
        self.execution_time = None

        self.start = tuple(zip(*np.where(maze == 'S')))[0]
        self.goal = tuple(zip(*np.where(maze == 'G')))[0]

        self.visited = np.zeros_like(maze, dtype=bool)
        self.solution_path = []

    def is_valid_move(self, x, y):
        """
        Check if the move is valid (within bounds and not a wall).

        Args:
        x (int): x-coordinate
        y (int): y-coordinate

        Returns:
        bool: True if move is valid, False otherwise.
        """
        return (0 <= x < self.width and
                0 <= y < self.height and
                self.maze[y, x] != '#' and
                not self.visited[y, x])

    def solve(self):
        """
        Solve the maze using Iterative Depth-First Search.

        Returns:
        list: Solution path if found, empty list otherwise.
        """
        # reset nodes explored, visited array, and solution path
        self.nodes_explored = 0
        self.visited = np.zeros_like(self.maze, dtype=bool)
        self.solution_path = []

        # define start point
        start_x, start_y = self.start

        # moves for dfs: down, right, up, left
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        stack = [(start_x, start_y, [])]

        while stack:
            current_x, current_y, path = stack.pop()

            if self.visited[current_y, current_x]:
                continue

            self.visited[current_y, current_x] = True
            self.nodes_explored += 1

            current_path = path + [(current_x, current_y)]

            if (current_x, current_y) == self.goal:
                self.solution_path = current_path
                return current_path

            for dx, dy in moves:
                next_x, next_y = current_x + dx, current_y + dy

                if self.is_valid_move(next_x, next_y):
                    stack.append((next_x, next_y, current_path))

        # return empty array when no solution found
        return []

    def get_performance_metrics(self):
        """
        Return performance of the DFS algorithm for the maze solution.
        """
        return {
            'path_length': len(self.solution_path),
            'nodes_explored': self.nodes_explored,
            'nodes_available': self.nodes_available,
            'execution_time': self.execution_time,
            'is_solution_found': bool(self.solution_path)
        }

class BFSSolver(MazeSolverBase):
    """
    Maze solver using Breadth-First Search.
    """
    def __init__(self, title, maze):
        """
        Initialize the maze solver with a given maze array.

        Args:
        maze (numpy.ndarray): 2D array representing the maze 
        ('#' for walls, '.' for paths, 'S' for start, 'G' for goal).
        """
        super().__init__(title, maze)

        self.nodes_explored = 0
        self.nodes_available = len(list(zip(*np.where(maze == '.'))))
        self.execution_time = None

        # find start and goal positions
        self.start = tuple(zip(*np.where(maze == 'S')))[0]
        self.goal = tuple(zip(*np.where(maze == 'G')))[0]

        self.visited = np.zeros_like(maze, dtype=bool)
        self.solution_path = []

    def is_valid_move(self, x, y):
        """
        Check if the move is valid (within bounds and not a wall).

        Args:
        x (int): x-coordinate
        y (int): y-coordinate

        Returns:
        bool: True if move is valid, False otherwise.
        """
        return (0 <= x < self.width and
                0 <= y < self.height and
                self.maze[y, x] != '#' and
                not self.visited[y, x])

    def solve(self):
        """
        Solve the maze using Breadth-First Search.

        Returns:
        list: Solution path if found, empty list otherwise.
        """
        # reset nodes explored, visited array, and solution path
        self.nodes_explored = 0
        self.visited = np.zeros_like(self.maze, dtype=bool)
        self.solution_path = []

        # define start point
        start_x, start_y = self.start

        # moves for bfs: down, right, up, left
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # use a queue for BFS (First-In-First-Out)
        queue = deque([(start_x, start_y, [])])
        self.visited[start_y, start_x] = True  # Mark start as visited
        self.nodes_explored += 1

        while queue:
            current_x, current_y, path = queue.popleft()  # Get the oldest element (FIFO)

            current_path = path + [(current_x, current_y)]

            # Check if goal is reached
            if (current_x, current_y) == self.goal:
                self.solution_path = current_path
                return current_path

            # Try all four directions
            for dx, dy in moves:
                next_x, next_y = current_x + dx, current_y + dy

                # If move is valid and cell not visited
                if self.is_valid_move(next_x, next_y):
                    queue.append((next_x, next_y, current_path))
                    self.visited[next_y, next_x] = True  # Mark as visited when added to queue
                    self.nodes_explored += 1

        # return empty array when no solution found
        return []

    def get_performance_metrics(self):
        """
        Return performance of the BFS algorithm for the maze solution.
        """
        return {
            'path_length': len(self.solution_path),
            'nodes_explored': self.nodes_explored,
            'nodes_available': self.nodes_available,
            'execution_time': self.execution_time,
            'is_solution_found': bool(self.solution_path)
        }
