"""
Module implementing the SynchronizedFactory mechanism.

Author: Kajetan Frąckowiak
Date: 2026

Description: This file contains the full implementation of the SynchronizedFactory mechanism.
"""

import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces

class SynchronizedFactory(ParallelEnv):
    def __init__(self, render_mode=None):
        self.grid_size = 10
        self.agents = ["agents_0", "agents_1", "agents_2"]
        self.possible_agents = self.agents[:]
        self.reset_keys = ["done"]

        self.action_spaces = {a: spaces.Discrete(5) for a in self.agents}
        self.observation_spaces = {
            a: spaces.Box(low=0, high=self.grid_size, shape=(8,), dtype=np.float32)
            for a in self.agents
        }

        self.rm_state = 0
        self.reset()

    def reset(self, seed=None, options=None):
        self.agent_positions = {
            "agents_0": np.array([0, 0]),
            "agents_1": np.array([0, 1]),
            "agents_2": np.array([1, 0]),
        }
        self.crate_position = np.array([5, 5])
        self.goal_position = np.array([9, 9])
        self.rm_state = 0
        self.step_count = 0
        return self._get_obs(), {a: {} for a in self.agents}

    def _get_obs(self):
        obs = {}
        for a in self.agents:
            obs[a] = np.concatenate([
                self.agent_positions[a],
                self.crate_position,
                self.goal_position,
                [float(self.rm_state), self.step_count / 200.0],
            ]).astype(np.float32)
        return obs

    def _action_to_move(self, action):
        return {
            0: np.array([0, 0]),   # stay
            1: np.array([0, 1]),   # up (Y+)
            2: np.array([0, -1]),  # down (Y-)
            3: np.array([-1, 0]),  # left (X-)
            4: np.array([1, 0]),   # right (X+)
        }[int(action)]

    def _calculate_crate_move(self, actions):
        if self.rm_state < 1:
            return None
        p0, p2 = self.agent_positions["agents_0"], self.agent_positions["agents_2"]
        c = self.crate_position
        a0, a2 = int(actions["agents_0"]), int(actions["agents_2"])

        def check_diag(posA, actA, posB, actB, relA, reqA, relB, reqB):
            return np.array_equal(posA-c, relA) and actA == reqA and \
                   np.array_equal(posB-c, relB) and actB == reqB

        # Pchanie proste
        if (np.array_equal(p0-c, [-1,0]) and np.array_equal(p2-c, [1,0])) or \
           (np.array_equal(p2-c, [-1,0]) and np.array_equal(p0-c, [1,0])):
            if a0 == 1 and a2 == 1: return np.array([0, 1])
            if a0 == 2 and a2 == 2: return np.array([0, -1])

        if (np.array_equal(p0-c, [0,1]) and np.array_equal(p2-c, [0,-1])) or \
           (np.array_equal(p2-c, [0,1]) and np.array_equal(p0-c, [0,-1])):
            if a0 == 4 and a2 == 4: return np.array([1, 0])
            if a0 == 3 and a2 == 3: return np.array([-1, 0])

        # Skosy
        if check_diag(p0, a0, p2, a2, [-1,0], 4, [0,-1], 1) or check_diag(p2, a2, p0, a0, [-1,0], 4, [0,-1], 1):
            return np.array([1, 1])
        if check_diag(p0, a0, p2, a2, [1,0], 3, [0,-1], 1) or check_diag(p2, a2, p0, a0, [1,0], 3, [0,-1], 1):
            return np.array([-1, 1])
        if check_diag(p0, a0, p2, a2, [-1,0], 4, [0,1], 2) or check_diag(p2, a2, p0, a0, [-1,0], 4, [0,1], 2):
            return np.array([1, -1])
        if check_diag(p0, a0, p2, a2, [1,0], 3, [0,1], 2) or check_diag(p2, a2, p0, a0, [1,0], 3, [0,1], 2):
            return np.array([-1, -1])
        return None

    def step(self, actions):
        self.step_count += 1
        crate_move = self._calculate_crate_move(actions)
        if crate_move is not None:
            new_crate_pos = self.crate_position + crate_move
            if self._is_valid(new_crate_pos):
                self.crate_position = new_crate_pos

        for a in self.agents:
            move = self._action_to_move(actions[a])
            new_pos = self.agent_positions[a] + move
            if self._is_valid(new_pos) and not np.array_equal(new_pos, self.crate_position):
                self.agent_positions[a] = new_pos
                
        reward = -0.01  # Small step penalty to encourage efficiency
        terminated = False
        if self.rm_state == 0 and self._check_gate_condition():
            self.rm_state = 1
            reward += 2.0
        elif self.rm_state == 1 and np.array_equal(self.crate_position, self.goal_position):
            self.rm_state = 2
            reward += 10.0
            terminated = True

        rewards = {a: reward for a in self.agents}
        terminations = {a: terminated for a in self.agents}
        truncations = {a: self.step_count >= 200 for a in self.agents}
        infos = {a: {"rm_state": self.rm_state} for a in self.agents}
        return self._get_obs(), rewards, terminations, truncations, infos

    def render(self):
        grid = np.full((self.grid_size, self.grid_size), ".")
        
        # Helper to convert Cartesian [x,y] to Grid [row, col]
        def to_grid(pos):
            return (self.grid_size - 1 - int(pos[1]), int(pos[0]))

        # Goal, Crate, Button (P)
        gy, gx = to_grid(self.goal_position)
        grid[gy, gx] = "G"
        cy, cx = to_grid(self.crate_position)
        grid[cy, cx] = "C"
        py, px = to_grid(np.array([2, 2]))
        grid[py, px] = "P"

        # Agents (0, 1, 2)
        for i, a in enumerate(self.agents):
            ay, ax = to_grid(self.agent_positions[a])
            grid[ay, ax] = str(i)

        print("\n" + "\n".join([" ".join(row) for row in grid]))
        print(f"RM: {self.rm_state} | Step: {self.step_count} | Crate: {self.crate_position}")

    def _is_valid(self, pos):
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def _check_gate_condition(self):
        return np.array_equal(self.agent_positions["agents_2"], np.array([2, 2]))