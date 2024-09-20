import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import torch.cuda.amp as amp
from torch.nn import functional as F

import numpy as np
from constants import Constants as C
import random
from collections import deque

from typing import Tuple
from torch.utils.tensorboard import SummaryWriter

from memory.buffer import PrioritizedReplayBuffer

writer = SummaryWriter()
class ScumModel(nn.Module):
    def __init__(self):
        ## Create a neural network with 2 layers of 512 neurons each 
        super().__init__()
        self.linear_nn = nn.Sequential(
            nn.Linear(C.NUMBER_OF_POSSIBLE_STATES, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Linear(256, C.NUMBER_OF_POSSIBLE_STATES), 
        )

    def forward(self, x):
        qs_prediction = self.linear_nn(x)
        return qs_prediction


class DQNAgent:
    def __init__(self, epsilon: float = 1.0, learning_rate: float = 0.001, discount: float = None, path: str = None) -> None:
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.load_state_dict(self.model.state_dict())

        if path is not None:
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint)
            self.target_model.load_state_dict(checkpoint)

        self.buffer =  PrioritizedReplayBuffer(buffer_size=C.REPLAY_MEMORY_SIZE, state_size=C.NUMBER_OF_POSSIBLE_STATES, action_size=1, alpha=0.7, beta=0.4)## creamos una deque con maxima longitud C.REPLAY_MEMORY
        self.epsilon = epsilon
        self.epsilon_min = C.MIN_EPSILON
        self.epsilon_decay = C.EPSILON_DECAY

        self.target_update_counter = 0
        
        self.scaler = amp.GradScaler()
        # self.criterion = nn.HuberLoss()
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.discount = C.DISCOUNT if discount is None else discount

    def create_model(self) -> ScumModel:
        model = ScumModel().to(C.DEVICE)
        return model
    
    def update_replay_memory(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        self.buffer.add(transition)
    
    @torch.no_grad()
    def predict(self, state: np.ndarray, target: bool = False) -> np.ndarray:
        if state.ndim == 1:
            state = state[np.newaxis, :]
        model = self.target_model if target else self.model
        prediction = model(state)
        return prediction


    # Trains main network every step during episode
    def train(self, agent_number: int, step: int, terminal_state: bool) -> None:
        self.model.train()

        # Start training only if certain number of samples is already saved
        if self.buffer.real_size < C.MIN_REPLAY_MEMORY_SIZE:
            return None, None
            
        ## Sample a minibatch from the replay memory
        minibatch, weights, tree_idxs = self.buffer.sample(C.BATCH_SIZE)
        current_states, actions, rewards, new_current_states, finishes = minibatch

        current_qs_list = self.predict(current_states, target=False)
        future_qs_list = self.predict(new_current_states, target=True)

        possible_actions = self._get_possible_actions(new_current_states, future_qs_list)
        new_q = self._calculate_new_q_values(rewards, finishes, possible_actions)

        current_qs = current_qs_list.clone()
        actions = actions.long() - torch.tensor(1)

        current_qs[torch.arange(C.BATCH_SIZE).long(), actions] = new_q.float()

        td_errors, tree_idxs = self._optimize_model(current_states, current_qs, agent_number, step, weights, tree_idxs)

        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > C.UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
        return td_errors, tree_idxs

    def _extract_tensors_from_minibatch(self, minibatch):
        current_states = torch.stack([transition[0].to(C.DEVICE) for transition in minibatch])
        actions = torch.tensor([int(transition[1] - torch.tensor(1)) for transition in minibatch], device=C.DEVICE, dtype=torch.int64)
        rewards = torch.tensor([transition[2] for transition in minibatch], device=C.DEVICE, dtype=torch.long)
        new_current_states = torch.stack([transition[3].to(C.DEVICE) for transition in minibatch])
        finishes = torch.tensor([int(transition[4]) for transition in minibatch], device=C.DEVICE, dtype=torch.float32)
        return current_states, actions, rewards, new_current_states, finishes

    def _get_possible_actions(self, new_current_states, future_qs_list):
        tensor_w_possible_actions = new_current_states * future_qs_list
        possible_actions = torch.where(tensor_w_possible_actions == 0, torch.tensor(float("-inf"), device=C.DEVICE), tensor_w_possible_actions)
        
        ## If no action is possible, set the possible action to the future qs value so no -inf is set
        indices = (torch.sum(tensor_w_possible_actions, dim=1) == 0)
        possible_actions[indices] = future_qs_list[indices].clone()
        return possible_actions

    def _calculate_new_q_values(self, rewards, finishes, possible_actions):
        return rewards + self.discount * (1 - finishes) * torch.max(possible_actions, dim=1).values

    def _optimize_model(self, batch_X, batch_y, agent_number, step, weights, tree_idxs):
        with amp.autocast():
            outputs = self.model(batch_X)
            td_errors = torch.mean(torch.abs(batch_y - outputs), dim=1).detach()
            loss = torch.mean((batch_y - outputs)**2 * weights)
            loss = torch.clip(loss, -1, 1)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            writer.add_scalar(f"Loss/Agent {agent_number}", loss, global_step=step)
            writer.flush()

        return td_errors, tree_idxs

    def decay_epsilon(self):
        """
        Decay the epsilon value for epsilon-greedy action selection.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

    def save_model(self, path: str = "model.pt") -> None:
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str = "model.pt") -> nn.Module:
        model = torch.load(path)
        return model
