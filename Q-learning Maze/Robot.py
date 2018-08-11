import random
import numpy as np
from collections import defaultdict
import math

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.alpha0 = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # TODO 1. No random choice when testing
            self.epsilon = 0
        else:
            # TODO 2. Update parameters when learning
            self.t += 1
            self.epsilon = self.epsilon0/(self.t/10 + 1)

            return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """

        # TODO 3. Return robot's current state
        return self.maze.sense_robot()

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # TODO 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        '''
        if state not in self.Qtable.keys():
            default = dict(zip(self.valid_actions,[0] * len(self.valid_actions)))
            self.Qtable[state] = default
        '''

        #Optimizate the code with defaultdict
        self.Qtable.setdefault(state, {a: 0.0 for a in self.valid_actions})

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():

            # TODO 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            '''
            x = random.uniform(0,1)

            if x < self.epsilon:
                return True
            else:
                return False
            '''
            #Optimizate the code after first review
            return random.uniform(0, 1) < self.epsilon

        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                action = random.choice(self.valid_actions)
                return action
            else:
                # TODO 7. Return action with highest q value
                sorted_actions = sorted(self.Qtable[self.state].items(), key=lambda x:x[1], reverse=True)
                max_action = sorted_actions[0][1]
                #construct a stochastic policy that puts equal probability on maximizing actions
                best_actions = []
                for a in sorted_actions:
                    if a[1] == max_action:
                        best_actions.append(a[0])
                #print(best_actions)
                return random.choice(best_actions)
                #best_a_index = np.argwhere(sorted_actions[:,1] == np.max(sorted_actions[:,1])).flatten()
                #actions = [sorted_actions[i,0] for i in best_a_index]
                #return sorted(self.Qtable[self.state].items(), key=lambda x:x[1], reverse=True)[0][0]
        elif self.testing:
            # TODO 7. choose action with highest q value
            return sorted(self.Qtable[self.state].items(), key=lambda x:x[1], reverse=True)[0][0]
        else:
            # TODO 6. Return random choose aciton
            return random.choice(self.valid_actions)

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
            '''
            #Optimizate the code after first review>>>
            self.epsilon = 0
            old_state = self.state
            self.state = next_state
            next_action = self.choose_action()
            '''
            old_state = self.state
            max_next_q = max(self.Qtable[next_state].values())
            self.Qtable[old_state][action] = (1.0 - self.alpha) * self.Qtable[old_state][action] + self.alpha * (r + self.gamma * max_next_q)
            #Optimizate the code after first review<<<
            # TODO 8. When learning, update the q table according
            # to the given rules

    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state() # Get the current state
        self.create_Qtable_line(self.state) # For the state, create q table line
        action = self.choose_action() # choose action for this state
        reward = self.maze.move_robot(action) # move robot for given action
        #print(action)

        next_state = self.sense_state() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state
        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state) # update q table
            self.update_parameter() # update parameters

        return action, reward
