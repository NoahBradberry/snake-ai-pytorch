import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, target_model, lr, gamma, tau):
        self.model = model
        self.target_model = target_model
        self.lr = lr
        self.gamma = gamma
        self.tau = tau

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        max_next_q = torch.max(next_q_values, dim=1)[0]
        target = q_values.clone()

        for i in range(len(dones)):
            if dones[i]:
                target[i][actions[i].argmax()] = rewards[i]
            else:
                target[i][actions[i].argmax()] = rewards[i] + self.gamma * max_next_q[i]

        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            for params, target_params in zip(self.model.parameters(), self.target_model.parameters()):
                target_params.data = (
                    self.tau * params.data + (1 - self.tau) * target_params.data
                )




