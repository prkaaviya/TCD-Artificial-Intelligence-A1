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
TEXT_DIR = os.path.join(MAZE_DIR, 'text')
VISUALS_DIR = os.path.join(RESULTS_DIR, 'visuals')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

def main():
    """
    The main entrypoint for running the maze solvers.
    """
    print("\n---BEGIN---\n")

    os.makedirs(VISUALS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Solve a maze using various search algorithms.")
    parser.add_argument('-T', '--title', type=str, default='maze1',
                        help="Title/name of the maze file under mazes/ (without .txt extension)")
    parser.add_argument('-A', '--algorithm', type=str, default='DFS',
                        choices=["DFS", "BFS", "A*", "MDP_VALUE", "MDP_POLICY"],
                        help="Algorithm to use for solving the maze")

    args = parser.parse_args()
    
    

    maze_path = f"{TEXT_DIR}/{args.title}.txt"
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

    if solution:
        print("Solution found!")
        print("Solution path:", solution)
        solver.visualize_solution(solution, f'{VISUALS_DIR}/{args.title}_solution.png')
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

    save_metrics(metrics_data, f'{METRICS_DIR}/{args.title}_metrics.csv')

    print("\n---DONE---\n")

if __name__ == "__main__":
    main()