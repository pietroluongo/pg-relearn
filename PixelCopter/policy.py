import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from numbers import Number
from utils import get_device
from torch.types import Number
from typing import Any
from torch.optim import Optimizer
from collections import deque
import numpy as np
import torch.utils.tensorboard as tb


class Policy(nn.Module):
    def __init__(self, state_sz, hidden_sz, action_sz):
        super(Policy, self).__init__()
        self._device = get_device()
        self._fc1 = nn.Linear(state_sz, hidden_sz)
        self._fc2 = nn.Linear(hidden_sz, action_sz)
        self._net = nn.Sequential(
            nn.Linear(state_sz, hidden_sz),
            nn.ReLU(),
            nn.Linear(hidden_sz, action_sz),
            nn.Softmax(dim=1),
        )
        self._goodnet = nn.Sequential(
            nn.Linear(state_sz, hidden_sz),
            nn.ReLU(),
            nn.Linear(hidden_sz, 2 * hidden_sz),
            nn.ReLU(),
            nn.Linear(hidden_sz * 2, action_sz),
            nn.Softmax(dim=1),
        )
        self.to(self._device)

    def forward(self, x) -> torch.Tensor:
        # proc = self._fc2(F.relu(self._fc1(x)))
        # return F.softmax(proc, dim=1)
        return self._goodnet(x)
        # return self._net(x)

    def act(self, state) -> tuple[Number, Any]:
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def reinforce(
        self,
        optimizer: Optimizer,
        n_training_episodes: int,
        max_t: int,
        gamma: float,
        env,
        print_every=10,
    ):
        tb_writer = tb.writer.SummaryWriter(comment="goodnet")
        # Help us to calculate the score during the training
        scores_deque = deque(maxlen=100)
        scores = []
        # Line 3 of pseudocode
        for i_episode in range(1, n_training_episodes + 1):
            saved_log_probs = []
            rewards = []
            state = env.reset()
            # Line 4 of pseudocode
            for t in range(max_t):
                action, log_prob = self.act(state)
                saved_log_probs.append(log_prob)
                state, reward, done, *_ = env.step(action)
                rewards.append(reward)
                if done:
                    break
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))

            # Line 6 of pseudocode: calculate the return
            returns = deque(maxlen=max_t)
            n_steps = len(rewards)

            for t in range(n_steps)[::-1]:
                disc_return_t = returns[0] if len(returns) > 0 else 0
                returns.appendleft(gamma * disc_return_t + rewards[t])

            ## standardization of the returns is employed to make training more stable
            eps = np.finfo(np.float32).eps.item()
            ## eps is the smallest representable float, which is
            # added to the standard deviation of the returns to avoid numerical instabilities
            returns = torch.tensor(returns)
            returns = (returns - returns.mean()) / (returns.std() + eps)

            # Line 7:
            policy_loss = []
            for log_prob, disc_return in zip(saved_log_probs, returns):
                policy_loss.append(-log_prob * disc_return)
            policy_loss = torch.cat(policy_loss).sum()

            # Line 8: PyTorch prefers gradient descent
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            if i_episode % print_every == 0:
                tb_writer.add_scalar(
                    "Train/score", np.mean(np.array(scores_deque)), i_episode
                )
                print(
                    "Episode {}\tAverage Score: {:.2f}".format(
                        i_episode, np.mean(np.array(scores_deque))
                    )
                )

        return scores

    def eval(self, env, max_steps, n_eval_episodes):
        """
        Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
        :param env: The evaluation environment
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param policy: The Reinforce agent
        """
        episode_rewards = []
        for episode in range(n_eval_episodes):
            state = env.reset()
            step = 0
            done = False
            total_rewards_ep = 0

            for step in range(max_steps):
                action, _ = self.act(state)
                new_state, reward, done, *_ = env.step(action)
                total_rewards_ep += reward
                if done:
                    break
                state = new_state
            episode_rewards.append(total_rewards_ep)
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)

        return mean_reward, std_reward
