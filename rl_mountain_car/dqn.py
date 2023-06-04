import random
from typing import Union

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam


class Buffer:
    """経験再生に使用する経験データ保持用のBufferクラス"""

    def __init__(self, buffer_size: int, batch_size: int) -> None:
        """
        Args:
            buffer_size: 格納しておく経験データ数
            batch_size: 一度に取り出すデータ数
        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(
        self,
        state: tuple[float, float],
        action: int,
        reward: Union[int, float],
        next_state: tuple[float, float],
        done: bool,
    ):
        """経験データをバッファに追加する
        Args:
            state: 現在の状態(mountain carでは、(位置, 速度))
            action: 行動(mountain carでは、0, 1, 2)
            reward: 報酬
            next_state: 1ステップ後の状態
            done: 終了したか
        """
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def get_batch(self):
        """バッファからランダムに経験データをバッチサイズ分取り出す(ランダムにすることでデータ間の相関を減らす)。
        Retrns:
            state: 現在の状態 shape = (B, 2)
            action: 行動 shape = (B, )
            reward: 報酬 shape = (B, )
            next_state: 1ステップ後の状態 shape = (B, 2)
            done: 終了したか shape = (B, )
        """
        data = random.sample(self.buffer, self.batch_size)

        state = torch.tensor(np.stack([x[0] for x in data]))
        action = torch.tensor(np.stack([x[1] for x in data]).astype(np.int32))
        reward = torch.tensor(np.stack([x[2] for x in data]).astype(np.float32))
        next_stage = torch.tensor(np.stack([x[3] for x in data]))
        done = torch.tensor(np.stack([x[4] for x in data]).astype(np.int32))
        return state, action, reward, next_stage, done


class QNet(nn.Module):
    def __init__(self, action_size=3) -> None:
        super().__init__()
        self.l1 = nn.Linear(2, 200)
        self.l2 = nn.Linear(200, 200)
        self.l3 = nn.Linear(200, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


class DQNAgent:
    """DQNの学習と行動の取得を実施する。流れは下記
    1. 学習対象のqnet, ラベルデータ生成用qnet_targetを生成
        経験データのnext_stateをQNetに入力する際に、出力が変動しないようにするため
    2. 行動をε-greedy法で選択(greedyの場合、行動は、Q関数(qnet)の出力の最大値とする。)
    3. バッファに含まれた経験データからqnetの重み更新を実施
    4. 一定の周期でqnetとqnet_targetの重みを同期させる
    """

    def __init__(
        self,
        env,
        gamma=0.98,
        lr=0.0005,
        epsilon=0.1,
        buffer_size=10000,
        batch_size=32,
        action_size=3,
        optimizer=None,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.action_size = action_size

        self.buffer = Buffer(buffer_size=buffer_size, batch_size=batch_size)
        self.qnet = QNet(self.env.action_space.n)
        self.qnet_target = QNet(self.env.action_space.n)
        self.qnet_target.eval()
        self.sync_qnet()
        self.optimizer = (
            Adam(self.qnet.parameters(), lr=self.lr) if optimizer is None else optimizer
        )
        self.loss_fn = nn.HuberLoss()

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.tensor(state[None, :])
            qs = self.qnet(state)
            return qs.argmax().item()

    def update(
        self,
        state: tuple[float, float],
        action: int,
        reward: int,
        next_state: tuple[float, float],
        done: bool,
    ):
        self.buffer.add(state, action, reward, next_state, done)
        if len(self.buffer) < self.batch_size:
            return
        state, action, reward, next_state, done = self.buffer.get_batch()
        qs = self.qnet(state)
        q = qs[np.arange(len(action)), action]
        next_qs = self.qnet_target(next_state)
        next_q = next_qs.max(1)[0]

        next_q.detach()
        target = reward + (1 - done) * self.gamma * next_q
        loss = self.loss_fn(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, dst: str):
        torch.save(self.qnet.state_dict(), dst)

    def load_model(self, src: str):
        self.qnet.load_state_dict(torch.load(src))

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())
