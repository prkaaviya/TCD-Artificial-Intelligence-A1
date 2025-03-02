# CS7IS2: Artificial Intelligence - Assignment 1

This repository contains the source code, data processing notebooks and results for the completion of Assignment 1 in the Artificial Intelligence course at Trinity College Dublin. 

Author: Kaaviya Paranji Ramkumar

## Maze Generation and Solving Algorithms

This project implements various search and MDP algorithms for maze solving, including DFS, BFS, A*, and MDP Value/Policy Iteration methods. It compares their performance across mazes of different sizes using multiple metrics.

### Project Structure

```
.
├── LICENSE
├── README.md
├── gen_maze.py              # Maze generator
├── mazes/                   # Maze files
│   ├── text/                # Text representation of mazes
│   └── visuals/             # Visual representation of mazes
├── notebooks/               # Jupyter notebooks for analysis
├── requirements.txt         # Project dependencies
├── results/                 # Solution results
│   ├── metrics/             # Performance metrics
│   └── visuals/             # Solution visualizations
├── run.py                   # Main entry point
├── Makefile                 # Build automation
└── solvers/                 # Algorithm implementations
    ├── __init__.py
    ├── base.py              # Base abstract solver class
    ├── informed.py          # A* algorithm
    ├── mdp.py               # MDP-based algorithms
    ├── uninformed.py        # DFS and BFS algorithms
    └── utils.py             # Utility functions
```

### Implemented Algorithms

1. **Depth-First Search (DFS)**: A uninformed search algorithm that explores as far as possible along each branch before backtracking.
2. **Breadth-First Search (BFS)**: A uninformed search algorithm that explores all neighbors at the present depth before moving to nodes at the next depth level.
3. **A* Search**: An informed search algorithm that uses a heuristic function to guide its path expansion.
4. **MDP Value Iteration**: A dynamic programming approach that computes the utility of each state.
5. **MDP Policy Iteration**: An algorithm that computes an optimal policy by iteratively improving an existing policy.

### Installation

1. Clone the repository:
```bash
git clone git@github.com:prkaaviya/TCD-Artificial-Intelligence-A1.git
cd TCD-Artificial-Intelligence-A1
```

2. Install dependencies:
```bash
make setup
```
or
```bash
pip install -r requirements.txt
```

### Usage

#### Generating Mazes

Generate a single maze with specific dimensions:
```bash
make generate_maze SIZE=15 MAZE_NAME=my_maze
```

Generate mazes of predefined sizes (7x7, 9x9, 15x15, etc.):
```bash
make generate_mazes
```

#### Solving Mazes

Solve a specific maze with a specific algorithm:
```bash
make solve MAZE=maze1 ALG=BFS
```

Solve all available mazes with a specific algorithm:
```bash
make solve_all_mazes ALG=A*
```

### Performance Metrics

The following metrics are collected for each algorithm:
- Path length
- Number of nodes explored
- Number of nodes available
- Execution time
- Memory usage
- Solution existence

Results are saved as CSV files in the `results/metrics/` directory.

### Visualization

Maze solutions are visualized and saved as images in the `results/visuals/` directory. These visualizations show:
- The maze layout
- The start and goal positions
- The solution path (if found)

### Analysis

The Jupyter notebook in the `notebooks/` directory provides comprehensive analysis and comparison of the algorithms' performance across different maze sizes and configurations.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgements

The maze generation algorithm is based on the Recursive Backtracker algorithm with enforced boundaries.