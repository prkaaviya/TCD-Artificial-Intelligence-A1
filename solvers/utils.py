"""
Utility functions for maze solvers.
"""

import csv
import numpy as np

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

def save_metrics(metrics_data, filename):
    """
    Save metrics to a CSV file.
    """
    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, metrics_data.keys())
        w.writeheader()
        w.writerow(metrics_data)
