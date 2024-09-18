import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
import torch.cuda.amp as amp

import numpy as np
from constants import Constants as C
import random
from collections import deque

from typing import Tuple

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
        if path is not None:
            self.model = self.load_model(path)       
        else:
            self.model = self.create_model()
            
        self.target_model = self.create_model()
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.replay_memory = deque(maxlen=C.REPLAY_MEMORY_SIZE) ## creamos una deque con maxima longitud C.REPLAY_MEMORY
        self.epsilon = epsilon
        self.epsilon_min = C.MIN_EPSILON
        self.epsilon_decay = C.EPSILON_DECAY

        self.target_update_counter = 0
        
        self.scaler = amp.GradScaler()
        self.criterion = nn.HuberLoss()
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.discount = C.DISCOUNT if discount is None else discount

    def create_model(self) -> ScumModel:
        model = ScumModel().to(C.DEVICE)
        return model
    
    def update_replay_memory(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        self.replay_memory.append(transition)
    
    @torch.no_grad()
    def predict(self, state: np.ndarray, target: bool = False) -> np.ndarray:
        if state.ndim == 1:
            state = state[np.newaxis, :]
        model = self.target_model if target else self.model
        prediction = model(state)
        return prediction


    # Trains main network every step during episode
    def train(self, terminal_state):
        self.model.train()

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < C.MIN_REPLAY_MEMORY_SIZE:
            return
            
        ## Sample a minibatch from the replay memory
        minibatch = random.sample(self.replay_memory, C.BATCH_SIZE)

        ## Predict the Q's value for the current states
        current_states = torch.stack([transition[0] for transition in minibatch]).to(C.DEVICE) 
        current_qs_list = self.predict(current_states, target=False)

        ## delete impossible actions
        tensor_w_possible_actions = current_states*current_qs_list
        current_qs_list = torch.where(tensor_w_possible_actions == 0, torch.tensor(float('-inf'), device=C.DEVICE), tensor_w_possible_actions)


        ## Predict the Q's value for the new current states with the target model, which is the one that is updated every C.UPDATE_TARGET_EVERY episodes
        new_current_state = torch.stack([transition[3] for transition in minibatch]).to(C.DEVICE)
        future_qs_list = self.predict(new_current_state, target=True)

        ## delete impossible actions
        tensor_w_possible_actions = new_current_state*future_qs_list
        future_qs_list = torch.where(tensor_w_possible_actions == 0, torch.tensor(float('-inf'), device=C.DEVICE), tensor_w_possible_actions)


        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, _, finish) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not finish:
                # that .item() does is to get the value of the tensor as a python number    
                max_future_q = torch.max(future_qs_list[index]).item() 
                new_q = reward + self.discount * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            ## We clone the tensor to avoid in place operations
            current_qs = current_qs_list[index].clone()
            current_qs[action-1] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
        
        ## Convert the lists to tensors
        batch_X = torch.stack(X).to(C.DEVICE)
        batch_y = torch.stack(y).to(C.DEVICE)

        with amp.autocast():
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            # Backward pass and optimize
            ## this is for the mixed precision training,
            # it is used to scale the loss and the gradients 
            # so that the training is more stable
            self.scaler.scale(loss).backward() 
            self.scaler.step(self.optimizer)
            # Update the scale for next iteration
            self.scaler.update()
            
        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > C.UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0

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
