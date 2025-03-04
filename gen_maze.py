"""
gen_maze.py - Generate a maze using Recursive Backtracker algorithm with enforced boundaries.
"""
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MAZE_DIR = 'mazes'
VISUALS_DIR = os.path.join(MAZE_DIR, 'visuals')
TEXT_DIR = os.path.join(MAZE_DIR, 'text')

class MazeGenerator:
    def __init__(self, width, height):
        """
        Initialize a maze with given dimensions - (width, height).
        """
        # ensure dimensions are odd for proper wall placement
        self.width = width if width % 2 == 1 else width + 1
        self.height = height if height % 2 == 1 else height + 1
        # initialize maze with all walls
        self.maze = np.full((self.height, self.width), '#', dtype=str)

    def add_multiple_paths(self, path_density=0.1):
        """
        Add multiple paths to the maze by randomly removing walls.
        
        Args:
            path_density: Float between 0 and 1 indicating how many walls to remove
            (higher values create more paths)
        """
        # only consider interior walls, not boundary walls
        interior_walls = []

        for i in range(2, self.height-2):
            for j in range(2, self.width-2):
                if self.maze[i, j] == '#':
                    # check if removing this wall can connect two path cells
                    neighbors = [
                        (i-1, j), (i+1, j),  # vertical neighbors
                        (i, j-1), (i, j+1)   # horizontal neighbors
                    ]

                    path_neighbors = sum(1 for ni, nj in neighbors 
                                        if 0 <= ni < self.height and 0 <= nj < self.width
                                        and self.maze[ni, nj] in ['.', 'S', 'G'])

                    # if wall has at least two path neighbors, consider removing it
                    if path_neighbors >= 2:
                        interior_walls.append((i, j))

        num_walls_to_remove = int(len(interior_walls) * path_density)

        # randomly select walls to remove
        walls_to_remove = np.random.choice(
            len(interior_walls),
            size=min(num_walls_to_remove, len(interior_walls)),
            replace=False
        )

        # remove the selected walls
        for idx in walls_to_remove:
            i, j = interior_walls[idx]
            self.maze[i, j] = '.'

    def generate_recursive_backtracker(self, path_density=0.1):
        """
        Generate maze using Recursive Backtracker algorithm with enforced boundary walls.
        """
        # initialize all cells as walls
        self.maze = np.full((self.height, self.width), '#', dtype=str)

        # create path cells but keep boundaries as walls
        for i in range(1, self.height-1, 2):
            for j in range(1, self.width-1, 2):
                self.maze[i, j] = '.'

        # create visited array matching the actual cell dimensions
        visited = np.zeros((self.height, self.width), dtype=bool)
        stack = []

        # start at (1,1)
        start_y, start_x = 1, 1
        visited[start_y, start_x] = True
        stack.append((start_y, start_x))

        while stack:
            current_y, current_x = stack[-1]

            # find unvisited neighbors
            neighbors = []
            # follow the priority order - down, up, left, right
            for dy, dx in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                new_y, new_x = current_y + dy, current_x + dx
                # check whether neighbor is within inner bounds and not visited
                if (1 <= new_y < self.height-1 and 
                    1 <= new_x < self.width-1 and 
                    not visited[new_y, new_x] and 
                    self.maze[new_y, new_x] == '.'):  # check if it's a path cell
                    neighbors.append((new_y, new_x))

            if neighbors:
                # choose a random neighbor
                next_y, next_x = neighbors[int(np.random.randint(len(neighbors)))]
                # remove wall between cells
                wall_y = (current_y + next_y) // 2
                wall_x = (current_x + next_x) // 2
                self.maze[wall_y, wall_x] = '.'
                # mark the cell as visited and add to stack
                visited[next_y, next_x] = True
                stack.append((next_y, next_x))
            else:
                # backtrack
                stack.pop()

        # set start and end points
        self.maze[1, 1] = 'S'  # start coords
        self.maze[self.height-2, self.width-2] = 'G'  # goal coords

        if path_density > 0:
            self.add_multiple_paths(path_density)

    def visualize(self, output_img, title="Random Maze"):
        """
        Visualize the maze using matplotlib.
        """
        _, ax = plt.subplots(figsize=(8, 8))

        for i in range(self.height):
            for j in range(self.width):
                if self.maze[i, j] == '#':  # check for wall coords
                    ax.add_patch(Rectangle((j, self.height-1-i), 1, 1,
                                        facecolor='black'))
                elif self.maze[i, j] == 'S':  # check for start coords
                    ax.add_patch(Rectangle((j, self.height-1-i), 1, 1,
                                        facecolor='green'))
                elif self.maze[i, j] == 'G':  # check for goal coords
                    ax.add_patch(Rectangle((j, self.height-1-i), 1, 1,
                                        facecolor='red'))
                else:  # remaining path coords
                    ax.add_patch(Rectangle((j, self.height-1-i), 1, 1,
                                        facecolor='white'))

        plt.title(title)
        plt.axis('equal')
        plt.axis('off')

        os.makedirs(VISUALS_DIR, exist_ok=True)

        plt.savefig(f'{VISUALS_DIR}/{output_img}.png', bbox_inches='tight', dpi=300)
        plt.close()

    def save_maze_to_file(self, output_file):
        """
        Save the maze array to a text file.
        """
        os.makedirs(TEXT_DIR, exist_ok=True)
        np.savetxt(f'{TEXT_DIR}/{output_file}.txt', self.maze, fmt='%s', delimiter=',')

    def get_maze(self):
        """
        Return the current maze array.
        """
        return self.maze.copy()

def main():
    parser = argparse.ArgumentParser(description="Generate a custom maze using Recursive \
                Backtracker algorithm with enforced boundaries.")

    parser.add_argument('-W', '--width', type=int, default=9,
            help='Width of the maze (default: 9, will be adjusted to odd)')
    parser.add_argument('-H', '--height', type=int, default=9,
            help='Height of the maze (default: 9, will be adjusted to odd)')
    parser.add_argument('-I', '--image', type=str, default='mazes/maze1_img.png',
            help='Output image file path (default: maze1_img.png)')
    parser.add_argument('-T', '--text', type=str, default='mazes/maze1_arr.txt',
            help='Output text file path (default: maze1_arr.txt)')

    args = parser.parse_args()

    maze = MazeGenerator(args.width, args.height)

    maze.generate_recursive_backtracker()

    print(maze.get_maze())

    maze.visualize(args.image, title=f"Maze with Boundaries of size ({args.width}, {args.height})")

    maze.save_maze_to_file(args.text)

    print("Maze visualization and array saved in 'mazes' directory.")

if __name__ == "__main__":
    main()
