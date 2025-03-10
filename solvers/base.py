"""
base.py - Base class for all maze solvers.
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class MazeSolverBase(ABC):
    """
    Abstract base class that all maze solvers inherit from.
    """

    def __init__(self, title, maze):
        """
        Initialize base attributes needed by all maze solvers.

        Args:
            title: Name/title of the maze
            maze: The maze to solve (NumPy array)
        """
        self.maze = maze
        self.title = title
        self.height, self.width = maze.shape

    def visualize_solution(self, solution_path, algorithm_name=None, output_file=None):
        """
        Visualize the maze solution.
        
        Args:
        solution_path (list): List of (x, y) coordinates forming the solution path.
        output_file (str, optional): Path to save the visualization.
        algorithm_name (str, optional): Name of the algorithm to display in the title.
                                        If None, uses the class name.
        """
        _, ax = plt.subplots(figsize=(8, 8))

        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == '#':  # wall
                    ax.add_patch(plt.Rectangle((j, self.height-1-i), 1, 1,
                                        facecolor='black'))
                elif self.maze[i, j] == 'S':  # start
                    ax.add_patch(plt.Rectangle((j, self.height-1-i), 1, 1,
                                        facecolor='green'))
                elif self.maze[i, j] == 'G':  # goal
                    ax.add_patch(plt.Rectangle((j, self.height-1-i), 1, 1,
                                        facecolor='red'))
                else:  # path
                    ax.add_patch(plt.Rectangle((j, self.height-1-i), 1, 1,
                                        facecolor='white'))

        # plot the solution path on top of the maze
        if solution_path:
            path_x = [x + 0.5 for x, _ in solution_path]
            path_y = [self.height-1-y + 0.5 for _, y in solution_path]
            plt.plot(path_x, path_y, color='blue', linewidth=2, marker='o',
                    markersize=5, markerfacecolor='yellow')

        # get algorithm name from class name if not provided
        if algorithm_name is None:
            algorithm_name = self.__class__.__name__.replace("Solver", "")

        plt.title(f'{self.title} ({self.height}, {self.width}) Solution with {algorithm_name}')
        plt.axis('equal')
        plt.axis('off')

        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    @abstractmethod
    def solve(self):
        """Solve the maze and return a path from start to goal."""

    @abstractmethod
    def get_performance_metrics(self):
        """Solve the maze and return a path from start to goal."""
