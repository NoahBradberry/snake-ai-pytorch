import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

# Hyperparameters
MAX_MEMORY = 200_000
BATCH_SIZE = 2048
LR = 0.001
GAMMA = 0.95
EPSILON_DECAY = 0.997
MIN_EPSILON = 0.01
TAU = 0.01  # soft update for target network

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 1.0
        self.gamma = GAMMA

        self.model = Linear_QNet(11, 512, 256, 3)
        self.target_model = Linear_QNet(11, 512, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.trainer = QTrainer(self.model, self.target_model, lr=LR, gamma=self.gamma, tau=TAU)

        self.memory = deque(maxlen=MAX_MEMORY)
        self.batch_size = BATCH_SIZE

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step([state], [action], [reward], [next_state], [done])

    def get_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float)
        self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            move = torch.argmax(self.model(state_tensor)).item()
        action = [0, 0, 0]
        action[move] = 1
        return action

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - game.block_size, head.y)
        point_r = Point(head.x + game.block_size, head.y)
        point_u = Point(head.x, head.y - game.block_size)
        point_d = Point(head.x, head.y + game.block_size)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < head.x,  # food left
            game.food.x > head.x,  # food right
            game.food.y < head.y,  # food up
            game.food.y > head.y   # food down
        ]

        return [int(x) for x in state]


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                record = score
                agent.model.save()
            print('Game', agent.n_games, 'Score', score, 'Record:', record)
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == '__main__':
    train()
