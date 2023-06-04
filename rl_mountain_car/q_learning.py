import numpy as np
from gym import Env


class QLearningAgent:
    def __init__(
        self, env: Env, num_bins=30, init_epsilon=1.0, gamma=0.98, alpha=0.05
    ) -> None:
        self.env = env
        self.state_shape = env.observation_space.shape
        self.state_high = env.observation_space.high
        self.state_low = env.observation_space.low
        self.action_size = env.action_space.n

        self.gamma = gamma
        self.alpha = alpha

        # 離散化した際の状態のbin幅
        self.bin_width = (self.state_high - self.state_low) / num_bins

        self.Q = np.zeros((num_bins + 1, num_bins + 1, self.action_size))

        self.set_epsilon(init_epsilon)

    def discretize(self, state: tuple[float, float]) -> tuple[int, int]:
        """状態を離散化する
        Args:
            state: 状態 (位置, 速度)
        Returns:

        """
        # 入力した状態と状態の最小値との差
        state_diff_from_low = state - self.state_low
        # binが何個分かを計算して、intにキャストすることでbinのインデックスを取得
        bin_index = (state_diff_from_low / self.bin_width).astype(int)
        return tuple(bin_index)

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_epsilon(self):
        return self.epsilon

    def get_action(self, state):
        state = self.discretize(state)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        state = self.discretize(state)
        next_state = self.discretize(next_state)
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = np.max(next_qs)

        target = reward + self.gamma * next_q_max
        self.Q[state][action] += self.alpha * (target - self.Q[state][action])

    def save_q(self, dst):
        np.save(dst, self.Q)

    def load_q(self, src):
        self.Q = np.load(src)
        self.bin_width = (self.state_high - self.state_low) / (len(self.Q) - 1)
