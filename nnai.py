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
    if len(files) == 0:
        last = 0
    else:
        last = max(int(x.split("_")[1].replace(".sav",'')) for x in files)
    return last

class NNAI():

    def __init__(self, filename=None, epsilon=None) -> None:

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
            return action

        _, action = self.maxMove(game_state, depth)
        self.add_state(game_state)
        return action
    
    def maxMove(self, state: mergeGame, depth) -> Tuple[float, Optional[Tuple]]:
        
        moves = state.findLegalMoves()

        if len(moves) == 0:
            return state.score
        elif depth <= 0:
            nnscore = self.inference(state)
            print(f"NNScore: {nnscore}")
            return state.score + nnscore
        else:
            bestVal = -math.inf
            bestMove = None
            for move in moves:
                val = self.averageMove(state.generateSuccessorBoard(move), depth-1)
                if val > bestVal:
                    bestVal = val
                    bestMove = move
        return (bestVal, bestMove)

    def averageMove(self, state: mergeGame, depth):
        pieces = state.allPossibleNextPieces()
        mysum = 0
        for piece in pieces:
            val = self.maxMove(state.generateBoardNextPiece(piece), depth)
            if type(val) is tuple:
                val , _ = val
            mysum += val
        return mysum / len(pieces)

    def add_state(self, state):
        self.current_state_scores.append(state.score)
        self.current_states.append(get_rep(state))
    
    def terminate_learn(self, state):
        for ex in range(len(self.current_states)):
            self.fly_training[0].append(self.current_states[ex])
            self.fly_training[1].append(state.score - self.current_state_scores[ex])
        self.current_states = []

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

    def fit(self, trainfilename=None, trainmulti=None, partial=True, layer_sizes=None):
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
                self.model = MLPRegressor(hidden_layer_sizes=(200,200,200))
            else:
                self.model = MLPRegressor(layer_sizes)

        if partial:
            self.model.partial_fit(X=training[0], y=training[1])
        else:
            self.model.fit(X=training[0], y=training[1])

    def load_model(self, filename='nnai.sav'):
        self.model = pickle.load(open(filename, 'rb'))

    def save_model(self, filename='nnai.sav'):
        pickle.dump(self.model, open(filename, 'wb'))

        
    