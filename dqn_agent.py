import os
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim


import numpy as np
from constants import Constants as C
import random
from collections import deque

from typing import Tuple

class ScumModel(nn.Module):
    def __init__(self):
        ## Create a neural network with 2 layers of 512 neurons each 
        super().__init__()
        self.softmax_nn = nn.Sequential(
            nn.Linear(C.NUMBER_OF_POSSIBLE_STATES, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, C.NUMBER_OF_POSSIBLE_STATES), ## +1 will be the pass statement
        )
        self.softmax = nn.Softmax(dim=-1)  # Change dim=1 to dim=-1

    def forward(self, x):
        x = self.softmax_nn(x)
        proba = self.softmax(x)
        return proba


class DQNAgent:
    def __init__(self, epsilon: float = 1.0) -> None:
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.replay_memory = deque(maxlen=C.REPLAY_MEMORY_SIZE) ## creamos una deque con maxima longitud C.REPLAY_MEMORY
        self.epsilon = epsilon
        self.epsilon_min = C.MIN_EPSILON
        self.epsilon_decay = C.EPSILON_DECAY

        self.target_update_counter = 0

    def create_model(self) -> ScumModel:
        model = ScumModel().to(C.DEVICE)
        return model
    
    def update_replay_memory(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        self.replay_memory.append(transition)
    
    @torch.no_grad()
    def predict(self, state: np.ndarray, target: bool = False) -> np.ndarray:
        if state.ndim == 1:
            state = state[np.newaxis, :]
        state = torch.from_numpy(state).float().to(C.DEVICE)
        model = self.target_model if target else self.model
        prediction = model(state)
        return prediction


       # Trains main network every step during episode
    def train(self, terminal_state):

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()

        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < C.MIN_REPLAY_MEMORY_SIZE:
            return
        
        minibatch = random.sample(self.replay_memory, C.BATCH_SIZE) ## sample from

        ## Probably TODO
        current_states = np.array([transition[0] for transition in minibatch]) 

        # Get current states from minibatch, then query NN model for Q values
        current_qs_list = self.predict(current_states, target=False)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.predict(new_current_states, target=True)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, new_current_state, finish) in enumerate(minibatch):

            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not finish:

                max_future_q = np.max(future_qs_list[index].cpu().detach().numpy())
                new_q = reward + C.DISCOUNT * max_future_q
            else:
                new_q = reward

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action-1] = new_q

            # And append to our training data
            X.append(torch.from_numpy(current_state).float().to(C.DEVICE))
            y.append(current_qs)
        

        running_loss = 0.0
        for batch_X, batch_y in zip(X, y):
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_X.unsqueeze(0))  # Add unsqueeze(0)
            loss = self.criterion(outputs, batch_y.unsqueeze(0))  # Add unsqueeze(0)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
        

        # Update target network counter every episode
        if terminal_state:
            self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        if self.target_update_counter > C.UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

    def decay_epsilon(self):
        """
        Decay the epsilon value for epsilon-greedy action selection.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

