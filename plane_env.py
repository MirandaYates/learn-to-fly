import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.toy_text import discrete

class PlaneEnv(discrete.DiscreteEnv):

    metadata = {
        'render.modes': ['human']
    }

    def _calculate_transition_prob(self, current, delta):
        new_position = np.array(current) + np.array([delta, 0]) # adds action to pitch
        new_position += np.array([0, new_position[0]-self.max_pitch - self.gravity])
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        is_done = new_position[1] - self.goal_altitude == 0
        # is_done = np.array_equal(new_position, [self.max_pitch, self.goal_altitude])
        return [(1.0, new_state, -abs(new_position[1] - self.goal_altitude), is_done)]

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.max_altitude - 1)
        coord[1] = max(coord[1], 0)
        return coord

    # returns the current pitch
    def get_pitch(self, state):
        return np.unravel_index(state, self.shape)[0]

    # returns the current altitude
    def get_pitch(self, state):
        return np.unravel_index(state, self.shape)[1]

    def __init__(self):
        self.min_altitude = 0
        self.max_altitude = 100
        self.goal_altitude = self.max_altitude/2
        self.max_pitch = 3
        self.gravity = 1

        # shape of environment
        self.shape = ((self.max_pitch * 2) + 1, (self.max_altitude - self.min_altitude + 1))
        
        # number of states
        nS = np.prod(self.shape)

        # number of actions
        nA = 3

        # is done
        self.done = False
        self.seed()

        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = { a : [] for a in range(nA) }
            P[s][0] = self._calculate_transition_prob(position, -1)
            P[s][1] = self._calculate_transition_prob(position, 0)
            P[s][2] = self._calculate_transition_prob(position, 1)

        # random start location
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((np.random.randint(0, self.max_pitch * 2), np.random.randint(self.min_altitude, self.max_altitude)), self.shape)] = 1.0

        # calls DiscreteEnv
        super(PlaneEnv, self).__init__(nS, nA, P, isd)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # same as super, sets starting location to random
    def reset(self):
        discrete.DiscreteEnv.reset(self)
        self.s = np.ravel_multi_index((np.random.randint(0, self.max_pitch * 2), np.random.randint(self.min_altitude, self.max_altitude)), self.shape)


    def close(self):
        if self.viewer:
            self.viewer.close()

        self.viewer = None