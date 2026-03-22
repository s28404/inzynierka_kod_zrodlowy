import numpy as np
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces

class SynchronizedFactory(ParallelEnv):
    def __init__(self, render_mode=None):
        self.grid_size = 10
        self.agents = ["agents_0", "agents_1", "agents_2"]
        self.possible_agents = self.agents[:]

        # Atrybuty dla TorchRL
        self.reset_keys = ["done"]

        # AKCJE (UKŁAD KARTEZJAŃSKI):
        # 0: stay, 1: up (Y+1), 2: down (Y-1), 3: left (X-1), 4: right (X+1)
        self.action_spaces = {a: spaces.Discrete(5) for a in self.agents}

        # Obs: [x, y, crate_x, crate_y, goal_x, goal_y, rm_state, step]
        self.observation_spaces = {
            a: spaces.Box(low=0, high=self.grid_size, shape=(8,), dtype=np.float32)
            for a in self.agents
        }

        self.rm_state = 0
        self.reset()

    def reset(self, seed=None, options=None):
        # Start: Roboty w lewym dolnym rogu (blisko 0,0)
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
            obs[a] = np.concatenate(
                [
                    self.agent_positions[a],
                    self.crate_position,
                    self.goal_position,
                    [float(self.rm_state), self.step_count / 200.0],
                ]
            ).astype(np.float32)
        return obs

    def _action_to_move(self, action):
        return {
            0: np.array([0, 0]),   # stay
            1: np.array([0, 1]),   # up (Y+)
            2: np.array([0, -1]),  # down (Y-)
            3: np.array([-1, 0]),  # left (X-)
            4: np.array([1, 0]),   # right (X+)
        }[action]

    def step(self, actions):
        self.step_count += 1

        # 1. Ruch robotów (z uwzględnieniem granic X i Y)
        for a in self.agents:
            move = self._action_to_move(actions[a])
            new_pos = self.agent_positions[a] + move
            
            if self._is_valid(new_pos) and not np.array_equal(new_pos, self.crate_position):
                self.agent_positions[a] = new_pos

        # 2. Strong Coupling (Push logic)
        if self._check_joint_push(actions):
            move = self._action_to_move(actions["agents_0"])
            new_crate_pos = self.crate_position + move

            if self._is_valid(new_crate_pos):
                self.crate_position = new_crate_pos
                
        reward = 0.0
        terminated = False

        # 3. Reward Machine
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
        # Tworzymy siatkę wizualną (Y to wiersze, X to kolumny)
        grid = np.full((self.grid_size, self.grid_size), ".")

        # UWAGA: Przy renderowaniu macierzy, Y=0 jest na górze, 
        # więc odwracamy indeksowanie Y, żeby Y=0 było na dole ekranu.
        def to_grid(pos):
            return (self.grid_size - 1 - pos[1], pos[0])

        # Cel, Skrzynia, Przycisk
        gy, gx = to_grid(self.goal_position)
        grid[gy, gx] = "G"
        
        cy, cx = to_grid(self.crate_position)
        grid[cy, cx] = "C"
        
        py, px = to_grid(np.array([2, 2]))
        grid[py, px] = "P"

        # Agenci
        for i, a in enumerate(self.agents):
            pos = self.agent_positions[a]
            ay, ax = to_grid(pos)
            grid[ay, ax] = str(i)

        print("\n" + "\n".join([" ".join(row) for row in grid]))
        print(f"RM State: {self.rm_state} | Step: {self.step_count} | Crate: {self.crate_position}")

    def _is_valid(self, pos):
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def _check_joint_push(self, actions):
        pushers = []
        for a in ["agents_0", "agents_2"]:
            dist = np.sum(np.abs(self.agent_positions[a] - self.crate_position))
            if dist == 1:
                move = self._action_to_move(actions[a])
                # Jeśli ruch prowadzi "w" skrzynię
                if np.array_equal(self.agent_positions[a] + move, self.crate_position):
                    pushers.append(actions[a])
        return len(pushers) >= 2 and len(set(pushers)) == 1

    def _check_gate_condition(self):
        # agents_2 na polu (2, 2)
        return np.array_equal(self.agent_positions["agents_2"], np.array([2, 2]))