import mergedMain
import nnai_pytorch
from nnai_pytorch import NNAIPyTorch
import copy
import statistics
import os
import pickle
import time
import torch

def run_default(n=10, k=10, epsilon=None, temperature=None, filename='nnai_torch.sav', outfilename='nnai_torch.sav',
                    epochs=1, adam_lr=.01, weight_decay=0, amsgrad=False):
    agent = NNAIPyTorch(filename=filename, epsilon=epsilon, temperature=temperature)
    total_scores = []
    for match in range(n):
        print(f"Match {match+1}")
        scores = []
        for _ in range(k):
            game = mergedMain.mergeGame()
            moves = game.findLegalMoves()
            while len(moves) > 0:
                move = agent.get_action(game, 1)
                game.playMove(move)
                moves = game.findLegalMoves()
            agent.terminate_learn(game)
            print(f"Score: {game.score}")
            scores.append(game.score)
        agent.save_fly_training()
        agent.fit(epochs=epochs, adam_lr=adam_lr, weight_decay=weight_decay, amsgrad=amsgrad)
        agent.save_model(filename=outfilename)
        avg_score = statistics.mean(scores)
        print(f"Avg match score: {avg_score}")
        total_scores.append(avg_score)
    avg_score = statistics.mean(total_scores)
    print(f"Avg total score: {avg_score}")
    return avg_score

def eval_model(model='nnai_torch.sav', look_at=75, games=100):
    agent = NNAIPyTorch(filename=model)  # No exploration

    # TODO: Implement scoring procedure to get R2
    """
    print("Model info:")
    print(f"TODO: Model info")
    print("Scoring...")
    tm = os.listdir("training_torch")
    try:
        tm.remove('.DS_Store')
    except ValueError:
        pass
    last = nnai_pytorch.get_last_training()
    trainmulti=["training_torch/" + x for x in tm if int(x.split("_")[1].split(".")[0])>=last-look_at]
    training = [[], []]
    for fname in trainmulti:
        train = pickle.load(open(fname, 'rb'))
        training[0].extend(train[0])
        training[1].extend(train[1])
    model_score = agent.model.score(training[0], training[1])
    print(f"Model score: {model_score}")
    """
    
    scores = []
    for i in range(games):
        if i % 20 == 0:
            print(f"Game #{i}")
        game = mergedMain.mergeGame()
        moves = game.findLegalMoves()
        while len(moves) > 0:
            move = agent.get_action(game, 1)
            game.playMove(move)
            moves = game.findLegalMoves()
        scores.append(game.score)

    avg = statistics.mean(scores)
    my_max = max(scores)
    print(f"Average performance: {avg}")
    print(f"High score: {my_max}")

def draw_board(board):
    for row in board:
        for num in row:
            print(f" {num}", end='', flush=True)
        print()

def observe(model='nnai_torch.sav', look_at=100):
    agent = NNAIPyTorch(filename=model)
    
    game = mergedMain.mergeGame()
    moves = game.findLegalMoves()
    while len(moves) > 0:
        move = agent.get_action(game, 1)
        game.playMove(move)
        
        draw_board(game.board)
        print("Piece: " + str(game.nextPiece))
        print("Score: " + str(game.score))
        print("Last move: " + str(move))
        print("Inference: " + str(agent.inference(game)))
        userin = input()
        if userin == 'q':
            return 0
        
        moves = game.findLegalMoves()
    print(f"Score: {game.score}")

if __name__ == '__main__':

    # Last: avg 184 --> 185 --> 223 --> 198 --> 215 --> 224 --> 228 --> 247 --> 244
    #       --> 262 --> 266 --> 247 --> 278 --> 258 --> 265 --> 274 --> 276 --> 248
    #       --> 250
    run_default(n=300, adam_lr=.0001)
    eval_model(games=200)
