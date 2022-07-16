import torch
import random
import numpy as np
from collections import deque
from game import Game, Point, Direction
from model import Linear_QNet, QTrainer
from plot import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]

        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right
            (dir_r and game.is_collision(point_d)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)),

            # danger left
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_d and game.is_collision(point_r)),

            # move direction - one true
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, gameover):
        self.memory.append( (state, action, reward, next_state, gameover) ) # will popleft if max memory reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns list of tuples
        else:
            mini_sample = self.memory # or take whole sample

        states, actions, rewards, next_states, gameovers = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

    def train_short_memory(self, state, action, reward, next_state, gameover):
        self.trainer.train_step(state, action, reward, next_state, gameover)

    def get_action(self, state):
        # random moves (exploration): tradeoff between exploration and exploitation (of model)
        self.epsilon = 80 - self.n_games
        intended_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            # take a random move
            move_index = random.randint(0,2)
            intended_move[move_index] = 1
        else:
            # move based on model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move_index = torch.argmax(prediction).item()
            intended_move[move_index] = 1
        return intended_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    best_score = 0
    agent = Agent()
    game = Game()

    # training loop
    while True:

        previous_state = agent.get_state(game)

        # game step
        intended_move = agent.get_action(previous_state)
        reward, gameover, score = game.play_step(intended_move)

        new_state = agent.get_state(game)

        # train short_memory - each game step
        agent.train_short_memory(previous_state, intended_move, reward, new_state, gameover)
        agent.remember(previous_state, intended_move, reward, new_state, gameover)

        if gameover:
            # train the long_memory (experienced replay)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > best_score:
                best_score = score
                agent.model.save()

            print('Game: ', agent.n_games, 'Score: ', score, 'Record: ', best_score)

            # plot result
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()