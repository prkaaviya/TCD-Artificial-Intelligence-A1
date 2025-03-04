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
        super().__init__(title, maze)
        
        self.nodes_available = len(list(zip(*np.where(maze == '.'))))

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
        return self.maze[y, x] == '#'

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

        # calculate Manhattan distance to goal for getting gradient reward
        goal_x, goal_y = self.goal
        next_x, next_y = next_state
        curr_x, curr_y = state

        curr_dist = abs(curr_x - goal_x) + abs(curr_y - goal_y)
        next_dist = abs(next_x - goal_x) + abs(next_y - goal_y)

        # if the next state is closer to goal, give bonus
        if next_dist < curr_dist:
            return -0.5

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
        print(f"Start position: {self.start}")
        print(f"Goal position: {self.goal}")

        states = self.get_states()
        actions = self.get_actions()

        # Initialize value function with goal having high value
        self.values = {state: 0 for state in states}
        self.values[self.goal] = 100  # Set goal value higher

        # Set properties for Value Iteration
        self.iterations = 0
        self.states_evaluated = 0

        for i in range(self.max_iterations):
            self.iterations += 1
            delta = 0

            # Update values for all states
            for state in states:
                self.states_evaluated += 1

                # Skip updating value if goal state
                if state == self.goal:
                    continue

                old_value = self.values[state]

                # Calculate new value using Bellman equation
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

        # Extract policy from value function
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
        visited = {self.start}  # Track visited states to prevent loops

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

            # Check for loops
            if next_state in visited:
                print(f"Warning: Loop detected in path at {next_state}")
                break
                
            visited.add(next_state)
            path.append(next_state)
            current = next_state

        return path

    def get_performance_metrics(self):
        """
        Return performance metrics for the solver.

        Returns:
            metrics: Dictionary of performance metrics
        """
        path = self.extract_path()
        is_solution_found = len(path) > 1 and path[-1] == self.goal

        return {
            'path_length': len(path) if is_solution_found else 0,
            'nodes_available': self.nodes_available,
            'iterations': self.iterations,
            'states_evaluated': self.states_evaluated,
            'execution_time': self.execution_time,
            'is_solution_found': is_solution_found
        }

class MDPPolicyIterationSolver(MazeSolverBase):
    """
    Maze solver using MDP Policy Iteration solver.
    """
    def __init__(self, title, maze, discount_factor=0.999,
        theta=0.0001, max_iterations=500, policy_eval_iterations=50):
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
        super().__init__(title, maze)

        self.nodes_available = len(list(zip(*np.where(maze == '.'))))

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
        return self.maze[y, x] == '#'

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
        Enhanced reward structure with goal-directed gradient.
        
        Args:
            state: Current state (x, y)
            next_state: Next state (x, y)
            
        Returns:
            reward: Reward for this transition
        """
        # Set high reward for reaching the goal
        if next_state == self.goal:
            return 100
        
        # Set penalty for hitting a wall
        if self.is_wall(*next_state):
            return -100
        
        # Calculate Manhattan distance to goal for gradient reward
        goal_x, goal_y = self.goal
        next_x, next_y = next_state
        curr_x, curr_y = state
        
        # Get distances to goal
        curr_dist = abs(curr_x - goal_x) + abs(curr_y - goal_y)
        next_dist = abs(next_x - goal_x) + abs(next_y - goal_y)
        
        # If next state is closer to goal, give bonus
        if next_dist < curr_dist:
            return -0.5  # Small step penalty but better than standard step
        
        # Standard step penalty
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
        Evaluate policy with convergence check.

        Args:
            policy: Current policy mapping states to actions
            states: List of all states
            actions: List of possible actions

        Returns:
            values: Dictionary mapping states to their values
        """
        values = {state: 0 for state in states}
        
        # Set goal state value higher to create gradient
        values[self.goal] = 100
        
        for _ in range(self.policy_eval_iterations):
            delta = 0
            for state in states:
                self.states_evaluated += 1
                
                # Skip evaluating if reached goal state
                if state == self.goal:
                    continue
                    
                old_value = values[state]
                action = policy[state]
                
                # If no action is defined for this state, skip it
                if action is None:
                    continue
                    
                new_value = 0
                for next_state in states:
                    prob = self.get_transition_prob(state, action, next_state)
                    
                    if prob > 0:
                        reward = self.get_reward(state, next_state)
                        new_value += prob * (reward + self.discount_factor * values[next_state])
                        
                values[state] = new_value
                delta = max(delta, abs(old_value - new_value))
                
            # Check for convergence
            if delta < self.theta:
                break
                
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
        print(f"Start position: {self.start}")
        print(f"Goal position: {self.goal}")

        states = self.get_states()
        actions = self.get_actions()

        # Initialize policy with goal-directed actions
        self.policy = {}
        for state in states:
            if state == self.goal:
                self.policy[state] = None
            else:
                # Get all valid actions
                valid_actions = []
                for action in actions:
                    x, y = state
                    dx, dy = action
                    next_state = (x + dx, y + dy)
                    if not self.is_wall(*next_state):
                        valid_actions.append(action)
                        
                if valid_actions:
                    # Use action that gets closest to goal if possible
                    goal_x, goal_y = self.goal
                    best_action = None
                    min_distance = float('inf')
                    
                    for action in valid_actions:
                        x, y = state
                        dx, dy = action
                        next_x, next_y = x + dx, y + dy
                        dist = abs(next_x - goal_x) + abs(next_y - goal_y)
                        
                        if dist < min_distance:
                            min_distance = dist
                            best_action = action
                            
                    self.policy[state] = best_action
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
        visited = {self.start}  # Track visited states to prevent loops

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
                
            # Check for loops
            if next_state in visited:
                print(f"Warning: Loop detected in path at {next_state}")
                break
                
            visited.add(next_state)
            path.append(next_state)
            current = next_state

        return path

    def get_performance_metrics(self):
        """
        Return performance metrics for the solver.
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        path = self.extract_path()
        is_solution_found = len(path) > 1 and path[-1] == self.goal
        
        return {
            'path_length': len(path) if is_solution_found else 0,
            'nodes_available': self.nodes_available,
            'iterations': self.iterations,
            'policy_changes': self.policy_changes,
            'states_evaluated': self.states_evaluated,
            'execution_time': self.execution_time,
            'is_solution_found': is_solution_found
        }
