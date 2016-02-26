import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

def debug(s):
    if False:
        print s

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    valid_actions = (None, 'forward', 'left', 'right')
    gamma = 0.5
    alpha = 0.9 # learning rate

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q_table = {}
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.total_reward = 0
        self.deadline = None
        # For agent's learning reporting:
        self.num_iterations = 0
        self.failure_tracking = {}
        self.total_penalty = 0
        self.total_wrong_action = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.total_reward = 0
        self.num_iterations += 1

    # exploration probability that decreases as we have more iterations
    def exploration_prob(self):
        # return 0.1
        # return 0.4 / self.num_iterations
        return 0.4 / math.log((self.num_iterations * 2)**2)

    def print_Q_table(self):
        print "--------- Q-table ---------"
        for s in self.Q_table:
            print "{}: {}".format(s, self.Q_table[s])
        print "------- end Q-table -------"

    def Q(self, state, action):
        if state in self.Q_table:
            return self.Q_table[state][action] if action in self.Q_table[state] else 0
        else:
            return 0

    def update_Q(self, state, action, reward, new_state):
        if state not in self.Q_table:
            self.Q_table[state] = {}
        next_Q = []
        for a in self.valid_actions:
            next_Q.append(self.Q(new_state, a))
        self.Q_table[state][action] = ((1-self.alpha) * self.Q(state, action)) + self.alpha * (reward + self.gamma * max(next_Q))

    def choose_action(self):
        # action = random.choice(self.valid_actions)
        action = None
        Q_vals = {} # The Q values for each action from this state
        for a in self.valid_actions:
            Q_vals[a] = self.Q(self.state, a)

        # action = self.max_Q_strategy(Q_vals)
        action = self.optimal_strategy(Q_vals)

        debug("State: {}".format(self.state))
        debug(Q_vals)
        debug("chosen action: {}".format(action))
        return action

    def optimal_strategy(self, Q_vals):
        action = 'undefined'
        # Explore new thing.
        for a, val in Q_vals.items():
            if val == 0:
                action = a
                break
        if action == 'undefined':
            if random.random() < self.exploration_prob():
                # randomly explore non best action just to make sure we really didn't like that action
                action = random.choice(self.valid_actions)
            else:
                action = max(Q_vals, key=Q_vals.get)
        return action

    def max_Q_strategy(self, Q_vals):
        return max(Q_vals, key=Q_vals.get)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        debug("Next waypoint: {}".format(self.next_waypoint))
        inputs = self.env.sense(self)
        self.deadline = self.env.get_deadline(self)

        # TODO: Update state
        # In ideal world where the right of way applies, use these 4 input as state
        # self.state = "{},{},{},{}".format(inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

        # In this simulated world, the only thing that matters are:
        # inputs['light'], and next_waypoint (see environment.py line 164-187)
        self.state = "{},{}".format(inputs['light'], self.next_waypoint)

        # TODO: Select action according to your policy
        action = self.choose_action()

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        if self.last_state != None:
            self.update_Q(self.last_state, self.last_action, self.last_reward, self.state)

        self.total_reward += reward
        self.last_state = self.state
        self.last_action = action
        self.last_reward = reward
        if reward < 0:
            self.total_penalty += 1
        elif reward < 1:
            self.total_wrong_action += 1

        print("deadline = {}, inputs = {}, direction = {}, action = {}, reward = {}".format(self.deadline, inputs, self.next_waypoint, action, reward))
        if self.deadline < 0:
            self.failure_tracking[self.num_iterations - 1] = self.deadline
        if reward > 2:
            # Reached destination within deadline
            print "Total reward: {}".format(self.total_reward)
            print "Remaining deadline: {}".format(self.deadline)
            self.print_Q_table()
            if self.num_iterations == N_TRIALS:
                print "=== Summary of learning ==="
                print "Total iterations: {}".format(N_TRIALS)
                print "Gamma: {}".format(self.gamma)
                print "Alpha: {}".format(self.alpha)
                print "Final Exploration vs Exploitation: {} : {}".format(self.exploration_prob() * 10, (1 - self.exploration_prob()) * 10)
                print "Total failures: {} times it did not reach destination within deadline".format(len(self.failure_tracking))
                print "Total penalties: {} times it got -1".format(self.total_penalty)
                print "Total wrong action: {} times it did not follow the direction (got 0.5)".format(self.total_wrong_action)
                print "Iteration where it does not meet deadline: {}".format(self.failure_tracking)

class BasicDrivingAgent(Agent):
    """A basic driving agent"""

    def __init__(self, env):
        super(BasicDrivingAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color

    def update(self, t):
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        action = random.choice([None, 'forward', 'left', 'right'])
        reward = self.env.act(self, action) # Execute action and get reward
        print "BasicDrivingAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


N_TRIALS = 120
def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    # a = e.create_agent(BasicDrivingAgent)  # create agent that randomly choose an action
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=N_TRIALS)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
