import mergedMain
import nnai
import copy
import statistics
import os
import pickle
import time

# Generate test data, fit, repeat
def run_default(n=10, k=10, epsilon=None, temperature=None, filename='nnai.sav', outfilename='nnai.sav'):
    agent = nnai.NNAI(filename=filename, epsilon=epsilon, temperature=temperature)
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
        agent.fit()
        agent.save_model(filename=outfilename)
        avg_score = statistics.mean(scores)
        print(f"Avg match score: {avg_score}")
        total_scores.append(avg_score)
    avg_score = statistics.mean(total_scores)
    print(f"Avg total score: {avg_score}")
    return avg_score

def draw_board(board):
    for row in board:
        for num in row:
            print(f" {num}", end='', flush=True)
        print()

# Observe nnai playing
def observe(model='nnai.sav', look_at=100):
    agent = nnai.NNAI(filename=model)  # No exploration

    print("Model info:")
    print(f"Layers: {agent.model.n_layers_}, N-iter: {agent.model.n_iter_}, Max iter: {agent.model.max_iter}")
    tm = os.listdir("training")
    tm.remove('.DS_Store')
    last = nnai.get_last_training()
    trainmulti=["training/" + x for x in tm if int(x.split("_")[1].split(".")[0])>=last-look_at]
    training = [[], []]
    for fname in trainmulti:
        train = pickle.load(open(fname, 'rb'))
        training[0].extend(train[0])
        training[1].extend(train[1])
    model_score = agent.model.score(training[0], training[1])
    print(f"Model score: {model_score}")
    
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

# Load in all test data generated from training directory,
# fit, and see what happens. Saves model to super_nnai.sav
def test_super(filename=None, max_iter=100, replace_max_iter=False, look_at=100, outfilename='super_nnai.sav'):
    agent = nnai.NNAI(filename=filename)
    if replace_max_iter:
        agent.model.max_iter = max_iter
    tm = os.listdir("training")
    tm.remove('.DS_Store')
    print("Fitting...")
    last = nnai.get_last_training()
    trainmulti=["training/" + x for x in tm if int(x.split("_")[1].split(".")[0])>=last-look_at]
    agent.fit(trainmulti=trainmulti, partial=False, layer_sizes=(200,200,200,200,200), max_iter=max_iter)
    agent.save_model(filename=outfilename)
    observe(model=outfilename)

def eval_model(model='nnai.sav', look_at=75, games=100):
    agent = nnai.NNAI(filename=model)  # No exploration

    print("Model info:")
    print(f"Layers: {agent.model.n_layers_}, N-iter: {agent.model.n_iter_}, Max iter: {agent.model.max_iter}")
    print("Scoring...")
    tm = os.listdir("training")
    tm.remove('.DS_Store')
    last = nnai.get_last_training()
    trainmulti=["training/" + x for x in tm if int(x.split("_")[1].split(".")[0])>=last-look_at]
    training = [[], []]
    for fname in trainmulti:
        train = pickle.load(open(fname, 'rb'))
        training[0].extend(train[0])
        training[1].extend(train[1])
    model_score = agent.model.score(training[0], training[1])
    print(f"Model score: {model_score}")
    
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

if __name__ == '__main__':

    # run_default(n=100, filename='nnai.sav', temperature=3)
    # run_default(n=100, filename='nnai.sav')
    # Note: Last was 423
    # Then 491
    # observe()

    eval_model(model='nnai.sav', look_at=75, games=200)

    """
    scores = []
    for _ in range(10):
        try:
            scores.append(run_default(epsilon=.30))
        except:
            print("There was an error")
            
    with open("scores.csv", "w") as fhand:
        fhand.write("Time,Score\n")
        for i, score in enumerate(scores):
            fhand.write(str(i) + "," + str(score) + "\n")

    # No randomness now
    print("Running with no randomness")
    run_default(epsilon=0)"""