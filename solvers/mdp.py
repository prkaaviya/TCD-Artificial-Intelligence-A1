""" mdp.py - MDP-based algorithms: Value Iteration and Policy Iteration """
import numpy as np
from solvers.base import *

class MDPValueIterationSolver(MazeSolverBase):
    """
    Maze solver using MDP Value Iteration.
    """
    def __init__(self, title, maze, discount_factor=0.9, theta=0.001, max_iterations=1000):
        """
        Initialize the MDP Value Iteration solver.

        Args:
            title: Name/title of the maze
            maze: The maze to solve (NumPy array)
            discount_factor: Discount factor for future rewards (gamma)
            theta: Threshold for determining value convergence
            max_iterations: Maximum number of iterations to perform
        """
        self.maze = maze
        self.title = title
        self.height, self.width = maze.shape

        self.nodes_explored = 0
        self.nodes_available = 0
        self.execution_time = None

        self.start = tuple(zip(*np.where(maze == 'S')))[0]
        self.goal = tuple(zip(*np.where(maze == 'G')))[0]

        self.discount_factor = discount_factor
        self.theta = theta
        self.max_iterations = max_iterations
        self.values = {}
        self.policy = {}
        self.iterations = 0
        self.states_evaluated = 0

        self.solution_path = []

    def is_wall(self, x, y):
        """Check if a cell is a wall."""
        # check if coordinates are within bounds
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        # check if the cell is a wall (represented by '#')
        return self.maze[x, y] == '#'

    def get_states(self):
        """Get all valid states (positions) in the maze."""
        states = []
        for y in range(self.height):
            for x in range(self.width):
                # include only non-wall cells as valid states
                if not self.is_wall(x, y):
                    states.append((x, y))
        return states

    def get_actions(self):
        """Define possible actions as cardinal directions."""
        return [
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1)    # right
        ]

    def get_reward(self, state, next_state):
        """
        Define rewards for transitions.
        
        Args:
            state: Current state (row, col)
            next_state: Next state (row, col)
            
        Returns:
            reward: Reward for this transition
        """
        # set reward = 100 for reaching the goal
        if next_state == self.goal:
            return 100

        # set penalty = -100 for hitting a wall
        if self.is_wall(*next_state):
            return -100

        # set small penalty for each step to encourage shorter paths
        return -1

    def get_transition_prob(self, state, action, next_state):
        """
        Get transition probability P(next_state | state, action).
        For a deterministic environment, this is 1 if next_state is the result of
        applying action to state, and 0 otherwise.
        """
        x, y = state
        dx, dy = action
        expected_next_state = (x + dx, y + dy)

        # if next_state is the expected result of the action, probability is 1
        if expected_next_state == next_state:
            # check if the move is valid (i.e. it doesn't hit a wall)
            if not self.is_wall(*expected_next_state):
                return 1.0

        # if we're expecting to hit a wall, we stay in the same place
        if self.is_wall(*expected_next_state) and state == next_state:
            return 1.0

        # otherwise probability is 0
        return 0.0

    def solve(self):
        """
        Solve the maze using Value Iteration.
        
        Returns:
            path: List of positions forming the path from start to goal
        """
        states = self.get_states()
        actions = self.get_actions()

        # initialize value function
        self.values = {state: 0 for state in states}

        # set properties for Value Iteration
        self.iterations = 0
        self.states_evaluated = 0

        for i in range(self.max_iterations):
            self.iterations += 1
            delta = 0

            # update values for all states
            for state in states:
                self.states_evaluated += 1
                self.nodes_explored += 1

                # skip updating value if goal state
                if state == self.goal:
                    continue

                old_value = self.values[state]

                # calculate new value using Bellman equation
                new_value = float('-inf')

                for action in actions:
                    action_value = 0
                    for next_state in states:
                        prob = self.get_transition_prob(state, action, next_state)

                        if prob > 0:
                            reward = self.get_reward(state, next_state)
                            action_value += prob * \
                                (reward + self.discount_factor * self.values[next_state])
                    if action_value > new_value:
                        new_value = action_value

                self.values[state] = new_value
                delta = max(delta, abs(old_value - new_value))

            if delta < self.theta:
                print(f"Value Iteration converged after {i+1} iterations")
                break

        # extract policy from value function
        self.policy = {}
        for state in states:
            if state == self.goal:
                self.policy[state] = None
                continue

            best_action = None
            best_value = float('-inf')

            for action in actions:
                action_value = 0

                for next_state in states:
                    prob = self.get_transition_prob(state, action, next_state)
                    if prob > 0:
                        reward = self.get_reward(state, next_state)
                        action_value += prob * \
                            (reward + self.discount_factor * self.values[next_state])

                if action_value > best_value:
                    best_value = action_value
                    best_action = action

            self.policy[state] = best_action

        path = self.extract_path()
        self.solution_path = path
        return path

    def extract_path(self):
        """
        Extract the path from start to goal using the computed policy.

        Returns:
            path: List of positions from start to goal
        """
        path = [self.start]
        current = self.start

        max_path_length = self.width * self.height

        while current != self.goal and len(path) < max_path_length:
            action = self.policy[current]

            if action is None:
                break

            x, y = current
            dx, dy = action
            next_state = (x + dx, y + dy)

            if self.is_wall(*next_state):
                break

            path.append(next_state)
            current = next_state

        return path

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

        plt.title(f'{self.title} ({self.height}, {self.width}) Solution with MDP Value Iteration')
        plt.axis('equal')
        plt.axis('off')

        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def get_performance_metrics(self):
        """
        Return performance metrics for the solver.

        Returns:
            metrics: Dictionary of performance metrics
        """
        return {
            'path_length': len(self.extract_path()) - 1,  # subtract 1 to get number of steps
            'iterations': self.iterations,
            'nodes_explored': self.nodes_explored,
            'nodes_available': self.nodes_available,
            'states_evaluated': self.states_evaluated,
            'execution_time': self.execution_time,
            'is_solution_found': bool(self.solution_path)
        }

class MDPPolicyIterationSolver(MazeSolverBase):
    """
    Maze solver using MDP Policy Iteration solver.
    """
    def __init__(self, title, maze, discount_factor=0.9,\
        theta=0.001, max_iterations=100, policy_eval_iterations=10):
        """
        Initialize the MDP Policy Iteration solver.

        Args:
            title: Name/title of the maze
            maze: The maze to solve (NumPy array)
            discount_factor: Discount factor for future rewards (gamma)
            theta: Threshold for determining value convergence
            max_iterations: Maximum number of policy iterations to perform
            policy_eval_iterations: Number of iterations for policy evaluation step
        """
        self.maze = maze
        self.title = title
        self.height, self.width = maze.shape

        self.nodes_explored = 0
        self.nodes_available = 0
        self.execution_time = None

        self.start = tuple(zip(*np.where(maze == 'S')))[0]
        self.goal = tuple(zip(*np.where(maze == 'G')))[0]

        self.discount_factor = discount_factor
        self.theta = theta
        self.max_iterations = max_iterations
        self.policy_eval_iterations = policy_eval_iterations
        self.values = {}
        self.policy = {}
        self.iterations = 0
        self.policy_changes = 0
        self.states_evaluated = 0

        self.solution_path = []

    def is_wall(self, x, y):
        """Check if a cell is a wall."""
        # check if coordinates are within bounds
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        # check if the cell is a wall (typically represented by '#')
        return self.maze[x, y] == '#'

    def get_states(self):
        """Get all valid states (positions) in the maze."""
        states = []
        for y in range(self.height):
            for x in range(self.width):
                # include only non-wall cells as valid states
                if not self.is_wall(x, y):
                    states.append((x, y))
        return states

    def get_actions(self):
        """Define possible actions as cardinal directions."""
        return [
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1)    # right
        ]

    def get_reward(self, state, next_state):
        """
        Define rewards for transitions.
        
        Args:
            state: Current state (row, col)
            next_state: Next state (row, col)
            
        Returns:
            reward: Reward for this transition
        """
        # set reward = 100 for reaching the goal
        if next_state == self.goal:
            return 100

        # set penalty = -100 for hitting a wall
        if self.is_wall(*next_state):
            return -100

        # set small penalty for each step to encourage shorter paths
        return -1

    def get_transition_prob(self, state, action, next_state):
        """
        Get transition probability P(next_state | state, action).
        For a deterministic environment, this is 1 if next_state is the result of
        applying action to state, and 0 otherwise.
        """
        x, y = state
        dx, dy = action
        expected_next_state = (x + dx, y + dy)

        # if next_state is the expected result of the action, probability is 1
        if expected_next_state == next_state:
            # check if the move is valid (i.e. it doesn't hit a wall)
            if not self.is_wall(*expected_next_state):
                return 1.0

        # if we're expecting to hit a wall, we stay in the same place
        if self.is_wall(*expected_next_state) and state == next_state:
            return 1.0

        # otherwise probability is 0
        return 0.0

    def policy_evaluation(self, policy, states, actions):
        """
        Evaluate a policy by computing its value function.

        Args:
            policy: Current policy mapping states to actions
            states: List of all states
            actions: List of possible actions

        Returns:
            values: Dictionary mapping states to their values
        """
        values = {state: 0 for state in states}

        # evaluate policy for specified number of iterations
        for _ in range(self.policy_eval_iterations):
            for state in states:
                self.states_evaluated += 1
                # skip evaluating if reached goal state
                if state == self.goal:
                    continue

                action = policy[state]
                # if no action is defined for this state, skip it
                if action is None:
                    continue

                new_value = 0

                for next_state in states:
                    prob = self.get_transition_prob(state, action, next_state)

                    if prob > 0:
                        reward = self.get_reward(state, next_state)
                        new_value += prob * (reward + self.discount_factor * values[next_state])

                values[state] = new_value

        return values

    def policy_improvement(self, values, states, actions):
        """
        Improve policy based on value function.

        Args:
            values: Current value function
            states: List of all states
            actions: List of possible actions

        Returns:
            policy: Improved policy
            is_stable: Whether the policy has stabilized
        """
        policy = {}
        is_stable = True

        for state in states:
            if state == self.goal:
                policy[state] = None
                continue

            old_action = self.policy.get(state)

            best_action = None
            best_value = float('-inf')

            for action in actions:
                action_value = 0

                for next_state in states:
                    prob = self.get_transition_prob(state, action, next_state)
                    if prob > 0:
                        reward = self.get_reward(state, next_state)
                        action_value += prob * (reward + self.discount_factor * values[next_state])

                if action_value > best_value:
                    best_value = action_value
                    best_action = action

            policy[state] = best_action

            # check if policy has changed
            if old_action != best_action:
                is_stable = False
                self.policy_changes += 1

        return policy, is_stable

    def solve(self):
        """
        Solve the maze using Policy Iteration.
        
        Returns:
            path: List of positions forming the path from start to goal
        """
        states = self.get_states()
        actions = self.get_actions()

        # initialize policy randomly
        self.policy = {}
        for state in states:
            if state == self.goal:
                self.policy[state] = None
            else:
                # choose a random action that doesn't lead to a wall
                valid_actions = []
                for action in actions:
                    x, y = state
                    dx, dy = action
                    next_state = (x + dx, y + dy)
                    if not self.is_wall(*next_state):
                        valid_actions.append(action)
                if valid_actions:
                    self.policy[state] = valid_actions[0]  # just take the first valid action
                else:
                    self.policy[state] = None

        # set properties for policy iteration
        self.iterations = 0
        self.policy_changes = 0
        self.states_evaluated = 0

        for i in range(self.max_iterations):
            self.iterations += 1
            self.nodes_explored += len(states)

            # one, do Policy Evaluation
            self.values = self.policy_evaluation(self.policy, states, actions)

            # two, do Policy Improvement
            new_policy, is_stable = self.policy_improvement(self.values, states, actions)
            self.policy = new_policy

            # now check if policy has stabilized
            if is_stable:
                print(f"Policy Iteration converged after {i+1} iterations")
                break

        path = self.extract_path()
        self.solution_path = path
        return path

    def extract_path(self):
        """
        Extract the path from start to goal using the computed policy.
        
        Returns:
            path: List of positions from start to goal
        """
        path = [self.start]
        current = self.start

        max_path_length = self.width * self.height

        while current != self.goal and len(path) < max_path_length:
            action = self.policy[current]

            if action is None:
                break

            x, y = current
            dx, dy = action
            next_state = (x + dx, y + dy)

            # check if the next state is valid
            if self.is_wall(*next_state):
                # this shouldn't happen with a valid policy
                break

            path.append(next_state)
            current = next_state

        return path

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

        plt.title(f'{self.title} ({self.height}, {self.width}) Solution with MDP Policy Iteration')
        plt.axis('equal')
        plt.axis('off')

        if output_file:
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

    def get_performance_metrics(self):
        """
        Return performance metrics for the solver.
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        return {
            'path_length': len(self.extract_path()) - 1,  # subtract 1 to get number of steps
            'iterations': self.iterations,
            'nodes_explored': self.nodes_explored,
            'nodes_available': self.nodes_available,
            'policy_changes': self.policy_changes,
            'states_evaluated': self.states_evaluated,
            'execution_time': self.execution_time,
            'is_solution_found': bool(self.solution_path)
        }
