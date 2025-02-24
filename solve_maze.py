"""
solve_maze.py - Solve a maze using Informed and Uninformed Search Algorithms.
"""
import csv
import time
import argparse
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt

MAZE_DIR = 'mazes'

class MazeSolver:
    def __init__(self, title, maze):
        """
        Initialize the maze solver with a given maze array.

        Args:
        maze (numpy.ndarray): 2D array representing the maze 
        ('#' for walls, '.' for paths, 'S' for start, 'G' for goal).
        """
        self.maze = maze
        self.title = title
        self.height, self.width = maze.shape

        self.nodes_explored = 0
        self.nodes_available = 0
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
        Solve the maze using Iterative Depth-First Search.

        Returns:
        list: Solution path if found, empty list otherwise.
        """
        # Reset nodes explored, visited array, and solution path
        self.nodes_explored = 0
        self.visited = np.zeros_like(self.maze, dtype=bool)
        self.solution_path = []

        # Start point
        start_x, start_y = self.start

        # Possible moves: down, right, up, left
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        # Stack to simulate recursion: (x, y, current_path)
        stack = [(start_x, start_y, [])]

        while stack:
            current_x, current_y, path = stack.pop()

            # Skip if already visited
            if self.visited[current_y, current_x]:
                continue

            # Mark as visited and increment nodes explored
            self.visited[current_y, current_x] = True
            self.nodes_explored += 1

            # Update current path
            current_path = path + [(current_x, current_y)]

            # Check if goal is reached
            if (current_x, current_y) == self.goal:
                self.solution_path = current_path
                return current_path

            # Try moves in specified order: down, right, up, left
            for dx, dy in moves:
                next_x, next_y = current_x + dx, current_y + dy

                # If move is valid and not visited
                if self.is_valid_move(next_x, next_y):
                    stack.append((next_x, next_y, current_path))

        # No solution found
        return []

    def visualize_solution(self, solution_path, output_file=None):
        """
        Visualize the maze solution.

        Args:
        solution_path (list): List of (x, y) coordinates forming the solution path.
        output_file (str, optional): Path to save the visualization.
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
                    self.nodes_available += 1
                elif self.maze[i, j] == 'G':  # goal
                    ax.add_patch(plt.Rectangle((j, self.height-1-i), 1, 1,
                                        facecolor='red'))
                    self.nodes_available += 1
                else:  # path
                    ax.add_patch(plt.Rectangle((j, self.height-1-i), 1, 1,
                                        facecolor='white'))
                    self.nodes_available += 1

        # plot the solution path
        if solution_path:
            path_x = [x + 0.5 for x, _ in solution_path]
            path_y = [self.height-1-y + 0.5 for _, y in solution_path]
            plt.plot(path_x, path_y, color='blue', linewidth=2, marker='o',
                    markersize=5, markerfacecolor='yellow')

        plt.title(f'{self.title} ({self.height}, {self.width}) Solution with DFS')
        plt.axis('equal')
        plt.axis('off')

        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

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

def load_maze_from_file(filename):
    """
    Load maze from a CSV-style text file.

    Args:
    filename (str): Path to the text file containing the maze

    Returns:
    numpy.ndarray: 2D array representing the maze
    """
    try:
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            maze_list = list(csv_reader)
            maze_array = np.array(maze_list)
            return maze_array
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        raise
    except Exception as e:
        print(f"Error loading maze from {filename}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Solve a maze using \
                Informed and Uninformed Search Algorithms.")

    parser.add_argument('-T', '--title', type=str, default='maze1',
                        help="Title to the maze file under mazes/.")

    args = parser.parse_args()

    maze = load_maze_from_file(f"{MAZE_DIR}/{args.title}.txt")

    solver = MazeSolver(args.title, maze)

    tracemalloc.start()
    start_time = time.time()
    solution = solver.solve()
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    solver.execution_time = end_time - start_time

    if solution:
        print("Solution found!")
        print("Solution path:", solution)
        solver.visualize_solution(solution, f'{MAZE_DIR}/{args.title}_solution.png')
    else:
        print("No solution found.")

    memory_info = {
        'current_memory': current,
        'peak_memory': peak,
        'current_memory_mb': current / (1024 * 1024),
        'peak_memory_mb': peak / (1024 * 1024)
    }

    maze_info = {
        'maze_title': args.title,
        'maze_algorithm': 'DFS',
        'maze_height': solver.height,
        'maze_width': solver.width
    }

    print(f"\nAll Metrics for the {args.title} Solution\n---")
    metrics_data = maze_info | solver.get_performance_metrics() | memory_info
    for key, value in metrics_data.items():
        print(f"{key}: {value}")

    metrics_path = f"{MAZE_DIR}/{args.title}_metrics.csv"
    print(f"Saving results to {metrics_path}...")
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, metrics_data.keys())
        w.writeheader()
        w.writerow(metrics_data)

if __name__ == "__main__":
    main()
