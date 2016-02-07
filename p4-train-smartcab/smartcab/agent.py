import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    valid_actions = (None, 'forward', 'left', 'right')
    gamma = 0.8

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q_table = {}
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.time = 0
        self.total_reward = 0

    def reset(self, destination=None):
        print "Total reward: {}".format(self.total_reward)
        print self.Q_table

        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.total_reward = 0

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
        self.Q_table[state][action] = reward + self.gamma * max(next_Q)

    def choose_action(self):
        # action = random.choice(self.valid_actions)
        action = None
        Q_vals = {} # The Q values for each action from this state
        for a in self.valid_actions:
            Q_vals[a] = self.Q(self.state, a)

        print self.state
        print Q_vals
        # if max(Q_vals.values()) == 0:
        #     action = random.choice(self.valid_actions)
        #     # action = self.next_waypoint
        # else:
        #     action = max(Q_vals, key=Q_vals.get)

        action = self.max_Q_strategy(Q_vals)
        print "chosen action: {}".format(action)
        return action

    def max_Q_strategy(self, Q_vals):
        return max(Q_vals, key=Q_vals.get)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        print "Next waypoint: {}".format(self.next_waypoint)
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = "{}-{}-{}".format(inputs['light'], inputs['oncoming'], inputs['left']) #, self.next_waypoint , deadline

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
        self.time += 1

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        if self.time % 50 == 0:
            print self.Q_table

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


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    # a = e.create_agent(BasicDrivingAgent)  # create agent
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.3)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
