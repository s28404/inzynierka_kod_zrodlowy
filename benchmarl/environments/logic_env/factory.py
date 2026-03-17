#  Copyright (c) 2026 Kajetan Frąckowiak, s28404
#
#  Projekt: Algorytm DEMIR dla SMACv2 i Custom Logic Environment
#  Polsko-Japońska Akademia Technik Komputerowych, Wydział Informatyki
#  Praca Inżynierska (2026)
#
#  Opis: Custom PettingZoo ParallelEnv dla zadania logicznego fabrycznego (SynchronizedFactory).
#  Cechy: Collaborative robotics (Strong Coupling), Reward Machine FSM, Grid-based world.

import numpy as np
import functools
from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces


class SynchronizedFactory(ParallelEnv):
    def __init__(self, render_mode=None):
        # PDDL (Unified Planning) - zakomentowane na razie
        # Używamy tylko dla wartości opisowej w pracy
        # Trening odbywa się poprzez PettingZoo ParallelEnv API

        self.grid_size = 10
        self.agents = ["agents_0", "agents_1", "agents_2"]
        self.possible_agents = self.agents[:]

        # Atrybuty potrzebne dla TorchRL compatibility
        self.reset_keys = ["done"]

        # 0: stay, 1: left, 2: right, 3: down, 4: up
        self.action_spaces = {a: spaces.Discrete(5) for a in self.agents}

        # Obs: [self_x, self_y, crate_x, crate_y, goal_x, goal_y, rm_state, normalized_step]
        self.observation_spaces = {
            a: spaces.Box(low=0, high=self.grid_size, shape=(8,), dtype=np.float32)
            for a in self.agents
        }

        self.rm_state = 0
        self.reset()

    def reset(self, seed=None, options=None):
        # Pozycje startowe: Roboty w lewym górnym rogu
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

    def step(self, actions):
        self.step_count += 1

        # 1. Ruch robotów
        for a in self.agents:
            move = self._action_to_move(actions[a])
            new_pos = self.agent_positions[a] + move
            # Sprawdzenie granic i kolizji ze skrzynią (nie można wejść "w" skrzynię bez pchania)
            if self._is_valid(new_pos) and not np.array_equal(
                new_pos, self.crate_position
            ):
                self.agent_positions[a] = new_pos

        # 2. Mechanizm Strong Coupling (agents_0 i agents_2 muszą pchać razem)
        push_reward = 0
        if self._check_joint_push(actions):
            old_dist = np.linalg.norm(self.crate_position - self.goal_position)
            move = self._action_to_move(actions["agents_0"])
            new_crate_pos = self.crate_position + move

            if self._is_valid(new_crate_pos):
                self.crate_position = new_crate_pos
                new_dist = np.linalg.norm(self.crate_position - self.goal_position)

                if new_dist < old_dist:
                    push_reward = 0.5  # Nagroda tylko za pchanie w dobrym kierunku

        reward = push_reward
        terminated = False

        # 3. Reward Machine Logic
        # Stan 0 -> 1: agents_2 musi być na polu [2,2]
        if self.rm_state == 0 and self._check_gate_condition():
            self.rm_state = 1
            reward += 2.0

        # Stan 1 -> 2: Skrzynia w celu (wymaga wcześniejszego otwarcia "bramy" w stanie 0)
        elif self.rm_state == 1 and np.array_equal(
            self.crate_position, self.goal_position
        ):
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

        # Oznaczamy cel
        grid[self.goal_position[0], self.goal_position[1]] = "G"

        # Oznaczamy skrzynię
        grid[self.crate_position[0], self.crate_position[1]] = "C"

        # Oznaczamy przycisk (gate condition)
        grid[2, 2] = "P"  # P jak Plate/Button

        # Oznaczamy agentów
        for i, a in enumerate(self.agents):
            pos = self.agent_positions[a]
            grid[pos[0], pos[1]] = str(i)

        print("\n" + "\n".join([" ".join(row) for row in grid]))
        print(f"RM State: {self.rm_state} | Step: {self.step_count}")

    def _is_valid(self, pos):
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    def _action_to_move(self, action):
        return {
            0: np.array([0, 0]),  # stay
            1: np.array([-1, 0]),  # up/left (zależnie od osi)
            2: np.array([1, 0]),  # down/right
            3: np.array([0, -1]),
            4: np.array([0, 1]),
        }[action]

    def _check_joint_push(self, actions):
        pushers = []
        # Współpraca agents_0 i agents_2
        for a in ["agents_0", "agents_2"]:
            # dystans Manhattanowski == 1 oznacza, że robot stoi obok skrzyni
            dist = np.sum(np.abs(self.agent_positions[a] - self.crate_position))
            if dist == 1:
                move = self._action_to_move(actions[a])
                # Jeśli po ruchu robot wszedłby w pole skrzyni -> znaczy, że pcha
                if np.array_equal(self.agent_positions[a] + move, self.crate_position):
                    pushers.append(actions[a])

        # Pchają, jeśli jest ich dwóch i wybrali ten sam kierunek (ten sam index akcji)
        return len(pushers) >= 2 and len(set(pushers)) == 1

    def _check_gate_condition(self):
        return np.array_equal(self.agent_positions["agents_2"], np.array([2, 2]))
