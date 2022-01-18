from torch.optim import optimizer
from mergedMain import mergeGame
import math
import pickle
import os
import numpy as np
import random
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision.models as models
import pickle

# Get state representation for NN
def get_rep(state):
    # Note: this code was revised to be more optimized
    board_rep = [0] * 8 * 5 * 5
    i = 0
    for row in state.board:
        for col in row:
            board_rep[i+col] = 1
            i += 8
    return board_rep

def get_last_training():
    files = os.listdir('training_torch')
    try:
        files.remove('.DS_Store')
    except ValueError:
        pass
    if len(files) == 0:
        last = 0
    else:
        last = max(int(x.split("_")[1].replace(".sav",'')) for x in files)
    return last

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(5*5*8, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 1),
        )

    def forward(self, x):
        prediction = self.linear_relu_stack(x)
        return prediction


class NNAIPyTorch():

    def __init__(self, filename=None, epsilon=None, temperature=None, speak=False):

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

        # Purely for debugging
        self.softmax_calls = 0
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.speak = speak
        if speak:
            print(f"Using {self.device} device")

    def get_action(self, game_state: mergeGame):

        # If epsilon is non-zero, sometimes randomly picks a move
        if self.epsilon is not None and self.epsilon != 0 and random.random() < self.epsilon:
            moves = game_state.findLegalMoves() 
            action = moves[random.randrange(0, len(moves))]
            self.add_state(game_state)
        # Temperature induces use of softmax
        elif self.temperature is None:
            action = self.max_move(game_state)
            self.add_state(game_state)
        else:
            action = self.softmax_move(game_state)
            self.add_state(game_state)
        return action

    def softmax_move(self, state: mergeGame):
        
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
        score_list = [math.e**(x/self.temperature) for x in score_list]
        total = sum(score_list)
        score_list = [x / total for x in score_list]
        probs = np.nan_to_num(np.array([x for x in score_list]))
        my_sum = sum(probs)
        if my_sum < 1.0:
            probs[0] += 1.0 - my_sum
        elif my_sum > 1.0:
            for i in range(len(probs)):
                if probs[i] > my_sum - 1.0:
                    probs[i] -= my_sum - 1.0
        if self.softmax_calls == 0:
            # Put a breakpoint here
            self.softmax_calls = self.softmax_calls + 1
        else:
            self.softmax_calls = (self.softmax_calls + 1) % 150
        try:
            action = mv_list[np.random.choice([i for i in range(len(mv_list))], p=probs)]
        except ValueError:
            print("VALUE ERROR")
            print("\a")
            action = mv_list[0]
        return action
    
    def max_move(self, state: mergeGame):
        
        moves = state.findLegalMoves()

        if len(moves) == 0:
            return None
        max_action = None
        max_score = -math.inf
        next_states = [state.generateSuccessorBoard(mv) for mv in moves]
        next_state_reps = [get_rep(state) for state in next_states]
        self.model.eval()
        if self.model is None:
            nnscores = [0 for _ in range(len(next_states))]
        else:
            nnscores = [x.item() for x in self.model(torch.tensor(next_state_reps).float())]
        
        for i in range(len(next_states)):
            total_score = nnscores[i] + next_states[i].score
            if total_score > max_score:
                max_action = moves[i]
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
            filename = "training_torch/train_" + str(x) + ".sav"
        pickle.dump(self.fly_training, open(filename, 'wb'))
        self.fly_training = [[], []]
    
    def inference(self, state: mergeGame):

        if self.model is None:
            return 0
        self.model.eval()
        res = self.model(torch.tensor(get_rep(state)).float())[0].item()
        return res

    def train(self, dataloader, model, loss_fn, optimizer):
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def fit(self, trainfilename=None, trainmulti=None, layer_sizes=None, epochs=1, batch_size=None, adam_lr=.01, weight_decay=0, amsgrad=False):
        # Organizing training data
        if trainmulti is not None:
            training = [[], []]
            for fname in trainmulti:
                train = pickle.load(open(fname, 'rb'))
                training[0].extend(train[0])
                training[1].extend(train[1])
        else:
            # By default, trains using last training data
            if trainfilename is None:
                trainfilename = "training_torch/train_" + str(get_last_training()) + ".sav"
            training = pickle.load(open(trainfilename, 'rb'))
        
        # Initializing model
        if self.model is None:
            if layer_sizes is None:
                self.model = NeuralNetwork().to(self.device)
            else:
                # TODO: Add neural network params
                self.model = NeuralNetwork().to(self.device)

        # Actual training
        """
        
        The difference between 'partial' and 'not partial' made more sense in scikit learn
        Here, it just determines epoch size and batch size
        
        """
        # TODO Don't hardcode optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=adam_lr, weight_decay=weight_decay, amsgrad=amsgrad)
        loss_func = torch.nn.MSELoss()

        x = torch.tensor(training[0]).float().to(self.device)
        y = torch.reshape(torch.tensor(training[1]), (-1,1)).float().to(self.device)
        x, y = Variable(x), Variable(y)
        torch_dataset = Data.TensorDataset(x, y)
        if batch_size is None:
            batch_size = min(200, len(torch_dataset))
        torch_dataloader = Data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)
        for e in range(1, 1+epochs):
            self.train(torch_dataloader, self.model, loss_func, optimizer)
            if self.speak:
                print(f"Epoch {e}/{epochs}")
            
    def load_model(self, filename='nnai_torch.sav'):
        self.model = torch.load(filename)

    def save_model(self, filename='nnai_torch.sav'):
        # pickle.dump(self.model, open(filename, 'wb'))
        torch.save(self.model, filename)

    