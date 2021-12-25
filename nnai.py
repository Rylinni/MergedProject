from typing import Optional, Tuple
from mergedMain import mergeGame
import math
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
import pickle
import os
import numpy as np
import random

# Get state representation for NN
def get_rep(state):
    board_rep = []
    for row in state.board:
        for col in row:
            square = [0 for _ in range(8)]
            square[col] = 1
            board_rep.extend(square)
    return board_rep

def get_last_training():
    files = os.listdir('training')
    files.remove('.DS_Store')
    if len(files) == 0:
        last = 0
    else:
        last = max(int(x.split("_")[1].replace(".sav",'')) for x in files)
    return last

class NNAI():

    def __init__(self, filename=None, epsilon=None, temperature=None) -> None:

        """

        Plan:

        - Simulate games
        - For each move, there will be a call to get_action, then add_state
        - When a game is over, terminate_learn is called with the actual points
        - ---> The current states with actual points get added to fly_training
        - ---> After x number of games, the data are moved into local storage
        - ---> After kx games, the data are fit to a model, which is stored

        """

        # Probability with which alg chooses random move
        self.epsilon = epsilon
        
        self.temperature = temperature

        if filename is None:
            self.model = None
        else:
            self.load_model(filename)
        
        # On the fly training data
        # Collect/save n data --> Fit n data --> ... --> Save model
        self.fly_training = [[], []]  # X = list of states, y = payoffs
        # Payoff: number of points gotten after seeing state (not including before)
        
        # All states reached in current play through
        self.current_states = []

        self.current_state_scores = []

    def get_action(self, game_state: mergeGame, depth):

        # If epsilon is non-zero, sometimes randomly picks a move
        if self.epsilon is not None and self.epsilon != 0 and random.random() < self.epsilon:
            moves = game_state.findLegalMoves() 
            action = moves[random.randrange(0, len(moves))]
            self.add_state(game_state)
        # Temperature induces use of softmax
        elif self.temperature is None:
            action = self.max_move(game_state, depth)
            self.add_state(game_state)
        else:
            action = self.softmax_move(game_state, depth)
            self.add_state(game_state)
        return action

    def softmax_move(self, state: mergeGame, depth):
        
        moves = state.findLegalMoves()

        if len(moves) == 0:
            return None
        mv_list = []
        score_list = []
        for mv in moves:
            next_state = state.generateSuccessorBoard(mv)
            nnscore = self.inference(next_state)
            total_score = nnscore + next_state.score
            mv_list.append(mv)
            score_list.append(total_score)
        # Normalize, apply softmax
        total = sum([math.e**(x/self.temperature) for x in score_list])
        score_list = [math.e**(x/self.temperature) / total for x in score_list]
        probs = [x for x in score_list]
        return mv_list[np.random.choice([i for i in range(len(mv_list))], p=probs)]
    
    def max_move(self, state: mergeGame, depth):
        
        moves = state.findLegalMoves()

        if len(moves) == 0:
            return None
        max_action = None
        max_score = -math.inf
        for mv in moves:
            next_state = state.generateSuccessorBoard(mv)
            nnscore = self.inference(next_state)
            total_score = nnscore + next_state.score
            if total_score > max_score:
                max_action = mv
                max_score = total_score
        return max_action

    def add_state(self, state):
        self.current_state_scores.append(state.score)
        self.current_states.append(get_rep(state))
    
    def terminate_learn(self, state):
        for ex in range(len(self.current_states)):
            self.fly_training[0].append(self.current_states[ex])
            self.fly_training[1].append(state.score - self.current_state_scores[ex])
        self.current_states = []
        self.current_state_scores = []

    def save_fly_training(self, filename=None):
        if filename is None:
            x = get_last_training() + 1
            filename = "training/train_" + str(x) + ".sav"
        pickle.dump(self.fly_training, open(filename, 'wb'))
        self.fly_training = [[], []]
    
    def inference(self, state: mergeGame):
        if self.model is None:
            return 0
        res = self.model.predict([get_rep(state)])[0]
        return res

    def fit(self, trainfilename=None, trainmulti=None, partial=True, layer_sizes=None, max_iter=200):
        if trainmulti is not None:
            training = [[], []]
            for fname in trainmulti:
                train = pickle.load(open(fname, 'rb'))
                training[0].extend(train[0])
                training[1].extend(train[1])
        else:
            # By default, trains using last training data
            if trainfilename is None:
                trainfilename = "training/train_" + str(get_last_training()) + ".sav"
            training = pickle.load(open(trainfilename, 'rb'))
        if self.model is None:
            if layer_sizes is None:
                self.model = MLPRegressor(hidden_layer_sizes=(500,500,500), learning_rate_init=.01)
            else:
                self.model = MLPRegressor(hidden_layer_sizes=layer_sizes, max_iter=max_iter)

        if partial:
            self.model.partial_fit(X=training[0], y=training[1])
        else:
            self.model.fit(X=training[0], y=training[1])

    def load_model(self, filename='nnai.sav'):
        self.model = pickle.load(open(filename, 'rb'))

    def save_model(self, filename='nnai.sav'):
        pickle.dump(self.model, open(filename, 'wb'))

    
        
    