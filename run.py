"""
Main entry point for running maze solvers.
"""

import os
import time
import argparse
import tracemalloc

from solvers import get_solver_class
from solvers.utils import load_maze_from_file, save_metrics

MAZE_DIR = 'mazes'
RESULTS_DIR = 'results'
VISUALS_DIR = os.path.join(RESULTS_DIR, 'visuals')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

def main():
    os.makedirs(VISUALS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Solve a maze using various search algorithms.")
    parser.add_argument('-T', '--title', type=str, default='maze1',
                        help="Title/name of the maze file under mazes/ (without .txt extension)")
    parser.add_argument('-A', '--algorithm', type=str, default='DFS',
                        choices=["DFS", "BFS", "A*", "MDP_VALUE", "MDP_POLICY"],
                        help="Algorithm to use for solving the maze")

    args = parser.parse_args()

    maze_path = f"{MAZE_DIR}/{args.title}.txt"
    print(f"Loading maze from {maze_path}...")
    maze = load_maze_from_file(maze_path)

    solver_class = get_solver_class(args.algorithm)
    solver = solver_class(args.title, maze)

    tracemalloc.start()
    start_time = time.time()
    solution = solver.solve()
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    solver.execution_time = end_time - start_time

    # Handle results
    # (rest of your code that handles visualization and metrics)

if __name__ == "__main__":
    main()