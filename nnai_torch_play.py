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
                move = agent.get_action(game)
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
            move = agent.get_action(game)
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
        move = agent.get_action(game)
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
    # Started with lr=.001, weight decay = .0001
    # Changed lr to .01 after 3rd
    # Changed weight decay to 0 after 4

    # Note: inference is 80 for everything after 4, signaling something wrong?
    # But then it was more reasonable after 5 (which was cut off early after 100)

    # After 7, lr changed to .001

    # After 8, epochs changed to 2

    # After 9, k changed to 20, adam_lr changed to .0005 (epochs kept at 2)

    # After 10, adam_lr changed to .00025, epochs moved to 4, k changed to 25

    # After 11, adam lr changed to .0001, epochs moved to 1, k changed to 20

    # After 12, adam lr changed to .0005

    # After 13, adam lr changed to .0001

    # After 14, adam lr to .001 - run twice, once with temp 4 then temp 0
    # Also, for temp 4, k=5, for temp 0, k=15 (n=100 in both)

    # After 15, alternated between temperature = 5 and no exploration

    # After 16, changed lr to .0005, used epsilon .2 for n=100 k=5 then epsilon = 0 for n=100, k=5

    # After 18, did epsilon .1 with lr = .0005 then epsilon 0 with lr .0005

    # After 19, changed epsilon to .15 and did same alternating runs, but for longer - 
    # n=300 k=20 then n=600, k=10, eval changed to 300 games

    # After 21, changed epsilon to .05 in first run (same lengths)

    # After 22, changed epsilon to .1, n=500 on first go, n=800 on second

    # After 23, epsilon changed to .15, sizes kept

    # After 24, n=250,k=40, n=250,k=50, games in eval changed to 400

    # After 26, set n=10 on a hunch that the model at the end of 25 was a fluke (after
    # evaluating the model many times, it was clear that the average was around 315- it got 303
    # the first time it ran). The average match score over those 10 matces (50 games each) was 377.
    # The final evaluation was run twice - first time it got 402, second time it got 381. 
    # This model was saved, and afterward lr was set to .0001 with n=250, k=50, epsilon = 0
    # (no epsilon exploration)
    
    #        1       2       3       4       5       6       7       8       9
    # Last: 181 --> 168 --> 173 --> 175 --> 192 --> 193 --> 205 --> 216 --> 215 -->
    #        10      11      12      13      14      15      16      17      18
    #       230 --> 219 --> 245 --> 231 --> 244 --> 252 --> 254 --> 258 --> 251 -->
    #        19      20      21      22      23      24      25      26      27
    #       266 --> 292 --> 304 --> 306 --> 318 --> 328 --> 353 --> 315 --> 402
    # run_default(n=250, k=40, adam_lr=.0005, epsilon=.15)
    # run_default(n=10, k=50, adam_lr=.0005)
    eval_model(games=400)
