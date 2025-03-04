
# Makefile for AI Assignment 1

CONDA_ENV = tf
CONDA_RUN = conda run -n $(CONDA_ENV)

MAZE_DIR = mazes/text
RESULTS_DIR = results
METRICS_DIR = $(RESULTS_DIR)/metrics
VISUALS_DIR = $(RESULTS_DIR)/visuals

MAZE_SIZES = 7 9 15 17 21 25 31 45 67 101

ALGORITHMS = DFS BFS A* MDP_VALUE MDP_POLICY

.PHONY: dirs
dirs:
	mkdir -p $(MAZE_DIR) $(METRICS_DIR) $(VISUALS_DIR)

# Generate a single maze with specified size
.PHONY: generate_maze
generate_maze: dirs
	$(CONDA_RUN) python gen_maze.py --width $(SIZE) --height $(SIZE) --text $(MAZE_NAME) --image $(MAZE_NAME)

# Generate a series of mazes with predefined sizes
.PHONY: generate_mazes
generate_mazes: dirs
	@for size in $(MAZE_SIZES); do \
		echo "Generating maze with size $${size}x$${size}"; \
		$(CONDA_RUN) python  gen_maze.py --width $$size --height $$size --text maze_$$size --image maze_$$size; \
	done

# Solve a single maze with a specific algorithm
.PHONY: solve
solve: dirs
	$(CONDA_RUN) python run.py --title $(MAZE) --algorithm $(ALG)

# Solve a single maze with all algorithm
.PHONY: solve_one_maze_all_algorithms
solve_one_maze_all_algorithms: dirs
	@for alg in $(ALGORITHMS); do \
		echo "Solving $(MAZE) with $$alg"; \
		$(CONDA_RUN) python run.py --title $(MAZE) --algorithm $$alg; \
	done; \

# Solve all mazes with a specific algorithm
.PHONY: solve_all_mazes
solve_all_mazes: dirs
	@for maze in $(shell ls $(MAZE_DIR) | grep -o "maze[0-9]*" | sort -u); do \
		echo "Solving $$maze with $(ALG)"; \
		$(CONDA_RUN) python run.py --title $$maze --algorithm $(ALG); \
	done

# Run all solvers on all mazes
.PHONY: benchmark
benchmark: dirs
	@for maze in $(shell ls $(MAZE_DIR) | grep -o "maze[0-9]*" | sort -u); do \
		for alg in $(ALGORITHMS); do \
			echo "Benchmarking $$maze with $$alg"; \
			$(CONDA_RUN) python run.py --title $$maze --algorithm $$alg; \
		done; \
	done

.PHONY: setup
setup:
	pip install -r requirements.txt

# Specify default target
.PHONY: default
default:
	@echo "Available targets:"
	@echo "  make generate_maze SIZE=<size> MAZE_NAME=<name>  		- Generate a single maze"
	@echo "  make generate_mazes                              		- Generate mazes of different sizes"
	@echo "  make solve MAZE=<maze_name> ALG=<algorithm>      		- Solve a specific maze with an algorithm"
	@echo "  make solve_one_maze_all_algorithms MAZE=<maze_name>    - Solve specific mazes with all algorithms"
	@echo "  make solve_all_mazes ALG=<algorithm>             		- Solve all mazes with specific algorithm"
	@echo "  make setup                                       		- Install dependencies"