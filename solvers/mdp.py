""" mdp.py - MDP-based algorithms: Value Iteration and Policy Iteration """
import numpy as np
from solvers.base import MazeSolverBase

class MDPValueIterationSolver(MazeSolverBase):
    """
    Maze solver using MDP Value Iteration.
    """
    def __init__(self, maze, title, discount_factor=0.9, theta=0.001, max_iterations=1000):
        """
        Initialize the MDP Value Iteration solver.

        Args:
            maze: The maze to solve
            discount_factor: Discount factor for future rewards (gamma)
            theta: Threshold for determining value convergence
            max_iterations: Maximum number of iterations to perform
        """
        self.maze = maze
        self.title = title
        self.height, self.width = maze.shape

        self.discount_factor = discount_factor
        self.theta = theta
        self.max_iterations = max_iterations
        self.values = {}
        self.policy = {}
        self.iterations = 0
        self.states_evaluated = 0

    def get_states(self):
        """Get all valid states (positions) in the maze."""
        states = []
        for row in range(self.height):
            for col in range(self.width):
                # include only non-wall cells as valid states
                if not self.is_wall(row, col):
                    states.append((row, col))
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
        if next_state == self.maze.goal:
            return 100

        # set penalty = -100 for hitting a wall
        if self.maze.is_wall(*next_state):
            return -100

        # set small penalty for each step to encourage shorter paths
        return -1

    def get_transition_prob(self, state, action, next_state):
        """
        Get transition probability P(next_state | state, action).
        For a deterministic environment, this is 1 if next_state is the result of
        applying action to state, and 0 otherwise.
        """
        row, col = state
        d_row, d_col = action
        expected_next_state = (row + d_row, col + d_col)

        # if next_state is the expected result of the action, probability is 1
        if expected_next_state == next_state:
            # check if the move is valid (i.e. it doesn't hit a wall)
            if not self.maze.is_wall(*expected_next_state):
                return 1.0

        # if we're expecting to hit a wall, we stay in the same place
        if self.maze.is_wall(*expected_next_state) and state == next_state:
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

                # skip updating value if goal state
                if state == self.maze.goal:
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
            if state == self.maze.goal:
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
        return path

    def extract_path(self):
        """
        Extract the path from start to goal using the computed policy.
        
        Returns:
            path: List of positions from start to goal
        """
        path = [self.maze.start]
        current = self.maze.start

        max_path_length = self.maze.width * self.maze.height

        while current != self.maze.goal and len(path) < max_path_length:
            action = self.policy[current]

            if action is None:
                break

            row, col = current
            d_row, d_col = action
            next_state = (row + d_row, col + d_col)

            if self.maze.is_wall(*next_state):
                # this shouldn't happen with a valid policy
                break

            path.append(next_state)
            current = next_state

        return path

    def get_performance_metrics(self):
        """
        Return performance metrics for the solver.
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        return {
            'iterations': self.iterations,
            'states_evaluated': self.states_evaluated,
            'path_length': len(self.extract_path()) - 1,  # subtract 1 to get number of steps
            'algorithm': 'MDP Value Iteration'
        }

class MDPPolicyIterationSolver(MazeSolverBase):
    """
    Maze solver using MDP Policy Iteration solver.
    """
    def __init__(self, maze, title, discount_factor=0.9,
                theta=0.001, max_iterations=100, policy_eval_iterations=10):
        """
        Initialize the MDP Policy Iteration solver.

        Args:
            maze: The maze to solve
            discount_factor: Discount factor for future rewards (gamma)
            theta: Threshold for determining value convergence
            max_iterations: Maximum number of policy iterations to perform
            policy_eval_iterations: Number of iterations for policy evaluation step
        """
        self.maze = maze
        self.title = title
        self.height, self.width = maze.shape

        self.discount_factor = discount_factor
        self.theta = theta
        self.max_iterations = max_iterations
        self.policy_eval_iterations = policy_eval_iterations
        self.values = {}
        self.policy = {}
        self.iterations = 0
        self.policy_changes = 0
        self.states_evaluated = 0

    def get_states(self):
        """Get all valid states (positions) in the maze."""
        states = []
        for row in range(self.height):
            for col in range(self.width):
                # include only non-wall cells as valid states
                if not self.is_wall(row, col):
                    states.append((row, col))
        return states

    def get_actions(self):
        """Define possible actions as cardinal directions."""
        return [
            (-1, 0),  # Up
            (1, 0),   # Down
            (0, -1),  # Left
            (0, 1)    # Right
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
        if next_state == self.maze.goal:
            return 100

        # set penalty = -100 for hitting a wall
        if self.maze.is_wall(*next_state):
            return -100

        # set small penalty for each step to encourage shorter paths
        return -1

    def get_transition_prob(self, state, action, next_state):
        """
        Get transition probability P(next_state | state, action).
        For a deterministic environment, this is 1 if next_state is the result of
        applying action to state, and 0 otherwise.
        """
        row, col = state
        d_row, d_col = action
        expected_next_state = (row + d_row, col + d_col)

        # if next_state is the expected result of the action, probability is 1
        if expected_next_state == next_state:
            # check if the move is valid (i.e. it doesn't hit a wall)
            if not self.maze.is_wall(*expected_next_state):
                return 1.0

        # if we're expecting to hit a wall, we stay in the same place
        if self.maze.is_wall(*expected_next_state) and state == next_state:
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
                if state == self.maze.goal:
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
            if state == self.maze.goal:
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
            if state == self.maze.goal:
                self.policy[state] = None
            else:
                # choose a random action that doesn't lead to a wall
                valid_actions = []
                for action in actions:
                    row, col = state
                    d_row, d_col = action
                    next_state = (row + d_row, col + d_col)

                    if not self.maze.is_wall(*next_state):
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
        return path

    def extract_path(self):
        """
        Extract the path from start to goal using the computed policy.
        
        Returns:
            path: List of positions from start to goal
        """
        path = [self.maze.start]
        current = self.maze.start

        max_path_length = self.width * self.height

        while current != self.goal and len(path) < max_path_length:
            action = self.policy[current]

            if action is None:
                break

            row, col = current
            d_row, d_col = action
            next_state = (row + d_row, col + d_col)

            # check if the next state is valid
            if self.is_wall(*next_state):
                # this shouldn't happen with a valid policy
                break

            path.append(next_state)
            current = next_state

        return path

    def get_performance_metrics(self):
        """
        Return performance metrics for the solver.
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        return {
            'iterations': self.iterations,
            'policy_changes': self.policy_changes,
            'states_evaluated': self.states_evaluated,
            'path_length': len(self.extract_path()) - 1,  # Subtract 1 to get number of steps
            'algorithm': 'MDP Policy Iteration'
        }
