"""
informed.py - Maze solver for informed search algorithms: A*.
"""

import heapq
import numpy as np
from solvers.base import *

class AStarSolver(MazeSolverBase):
    """
    Maze solver using A* Search.
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

        self.counter = 0

        self.start = tuple(zip(*np.where(maze == 'S')))[0]
        self.goal = tuple(zip(*np.where(maze == 'G')))[0]

        self.visited = np.zeros_like(maze, dtype=bool)
        self.solution_path = []

    def manhattan_distance(self, pos1, pos2):
        """
        Calculate Manhattan distance heuristic.

        Args:
        pos1 (tuple): First position (x, y)
        pos2 (tuple): Second position (x, y)

        Returns:
        int: Manhattan distance between positions
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

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
        Solve the maze using A* Search.

        Returns:
        list: Solution path if found, empty list otherwise.
        """    
        self.nodes_explored = 0
        self.visited = np.zeros_like(self.maze, dtype=bool)
        self.solution_path = []

        # track maximum fringe size for metrics
        max_fringe_size = 0

        start_pos = self.start
        goal_pos = self.goal

        parent = {}

        g_score = {start_pos: 0}

        # initialize priority queue for open set
        # with format: (f_score, counter, position)
        open_set = [(self.manhattan_distance(start_pos, goal_pos), self.counter, start_pos)]
        self.counter += 1

        while open_set:
            # update max fringe size
            max_fringe_size = max(max_fringe_size, len(open_set))

            # get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            current_x, current_y = current

            # skip if already visited
            if self.visited[current_y, current_x]:
                continue

            # mark as visited and update explored nodes
            self.visited[current_y, current_x] = True
            self.nodes_explored += 1

            if current == goal_pos:
                # reconstruct path if goal reached
                self.solution_path = self._reconstruct_path(parent, current)
                return self.solution_path

            # explore neighbors in the order: down, right, up, left
            moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

            for dx, dy in moves:
                next_x, next_y = current_x + dx, current_y + dy
                next_pos = (next_x, next_y)

                if self.is_valid_move(next_x, next_y):
                    # calculate tentative g-score
                    tentative_g_score = g_score[current] + 1

                    # when we find a better path to this neighbor
                    if next_pos not in g_score or tentative_g_score < g_score[next_pos]:
                        # update the path
                        parent[next_pos] = current
                        g_score[next_pos] = tentative_g_score

                        # calculate f_score = g_score + heuristic
                        f_score = tentative_g_score + self.manhattan_distance(next_pos, goal_pos)
                        heapq.heappush(open_set, (f_score, self.counter, next_pos))
                        self.counter += 1

        # return empty array when no path found
        return []

    def _reconstruct_path(self, parent, current):
        """
        Reconstruct the path from start to goal.

        Args:
        parent (dict): Dictionary mapping node to its parent
        current (tuple): Current node (the goal)

        Returns:
        list: List of positions from start to goal
        """
        path = [current]
        while current in parent:
            current = parent[current]
            path.insert(0, current)
        return path

    def get_performance_metrics(self):
        """
        Return performance of the A* algorithm for the maze solution.
        """
        return {
            'path_length': len(self.solution_path),
            'nodes_explored': self.nodes_explored,
            'nodes_available': self.nodes_available,
            'execution_time': self.execution_time,
            'is_solution_found': bool(self.solution_path)
        }
