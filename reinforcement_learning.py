
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from datetime import datetime
from model import Net
from utilities import ExperienceMemory

SEED = random.seed(datetime.now())
HIDDEN_UNITES = 50
LEARNING_RATE = 0.0001
# CRITIC_LEARNING_RATE = 0.001
WEIGEHT_DECAY_RATE = 0
BATCH_SIZE = 128
GAMMA = 0.98
TAU = 1e-3
EPSILON = 0.9
TARGET_REPLACE_ITER = 100   # target update frequency
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DQN(object):
    def __init__(self, action_dim, state_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        # print(action_dim)
        # print(state_dim)
        self.memory = ExperienceMemory(action_dim, state_dim)
        self.learn_step_counter = 0
        if os.path.isfile('model.pkl'):
            self.load_model()
        else:
            self.model_evaluate = Net(state_dim, action_dim, HIDDEN_UNITES)#.to(DEVICE)
            self.model_target = Net(state_dim, action_dim, HIDDEN_UNITES)#.to(DEVICE)
            self.model_evaluate.double()
            self.model_target.double()
        self.model_optimizer = optim.Adam(self.model_evaluate.parameters(), lr=LEARNING_RATE)
        self.loss_funcion = nn.MSELoss()


    def select_action(self, current_state, add_noise=True):
        current_state = torch.from_numpy(current_state)#.float().to(DEVICE)
        # self.model_online.eval()
        # with torch.no_grad():
        if np.random.uniform() < EPSILON:
            # print(current_state)
            # print(current_state.unsqueeze(0))
            # actions_value = self.model_evaluate.forward(current_state)
            actions_value = self.model_evaluate.forward(current_state.unsqueeze(0))#.cpu().data.numpy()
            # print(current_state.unsqueeze(0).dim())
            # print(current_state.unsqueeze(0).size())
            # print("actions_value")
            # print(actions_value)
            action = torch.argmax(actions_value).data.numpy()
            # print("action")
            # print(action)
        else:
            action = np.random.randint(0, self.action_dim)
        return action

    def is_memory_full(self):
        return self.memory.is_memory_full()

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.model_target.load_state_dict(self.model_evaluate.state_dict())
        self.learn_step_counter += 1
        print("learn count: ", self.learn_step_counter)
        current_states, actions, rewards, next_states = self.memory.get_experiences(BATCH_SIZE)

        # print(current_states.dim())
        # print(current_states.size())

        # q_evaluate = self.model_evaluate(current_states, actions)
        # print(current_states)
        # print(actions)
        # q_evaluate = self.model_evaluate(current_states).gather(1, actions.type(torch.int64))
        # q_evaluate = self.model_evaluate(current_states)
        q_evaluate = self.model_evaluate.forward(current_states)
        # q_evaluate = self.model_evaluate(current_states, actions)
        # q_next = self.model_target(next_states)
        q_next = self.model_target.forward(next_states)
        q_target = rewards + GAMMA * q_next

        model_loss = F.mse_loss(q_evaluate, q_target) # TD error
        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()
        print("model loss: ", model_loss.data.numpy())
        return model_loss

    def store_experience(self, current_state, action, reward, next_state):
        self.memory.store_experience(current_state, action, reward, next_state)

    def save_model(self):
        torch.save(self.model_evaluate, 'model.pkl')

    def load_model(self):
        self.model_evaluate = torch.load('model.pkl').double()
        self.model_target = torch.load('model.pkl').double()

    # def save_critic(self):
    #     torch.save(self.critic_online, 'critic.pkl')

    # def load_critic(self):
    #     self.critic_online = torch.load('critic.pkl').double()
    #     self.critic_target = torch.load('critic.pkl').double()

