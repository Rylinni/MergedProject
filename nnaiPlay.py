import mergedMain
import nnai
import copy
import statistics
import os
import pickle

# Generate test data, fit, repeat
def run_default(n=10, k=10, epsilon=.05, filename='nnai.sav'):
    agent = nnai.NNAI(filename=filename, epsilon=epsilon)
    total_scores = []
    for match in range(n):
        print(f"Match {match+1}")
        scores = []
        for games in range(k):
            game = mergedMain.mergeGame()
            moves = game.findLegalMoves()
            while len(moves) > 0:
                gameCopy = copy.deepcopy(game)
                move = agent.get_action(gameCopy, 1)
                game.playMove(move)
                moves = game.findLegalMoves()
            agent.terminate_learn(game)
            print(f"Score: {game.score}")
            scores.append(game.score)
        agent.save_fly_training()
        agent.fit()
        agent.save_model()
        avg_score = statistics.mean(scores)
        print(f"Avg match score: {avg_score}")
        total_scores.append(avg_score)
    avg_score = statistics.mean(total_scores)
    print(f"Avg total score: {avg_score}")
    return avg_score

# Load in all test data generated from training directory,
# fit, and see what happens. Saves model to super_nnai.sav
def test_super():
    agent = nnai.NNAI()
    tm = os.listdir("training")
    tm.remove('.DS_Store')
    print("Fitting...")
    agent.fit(trainmulti=["training/" + x for x in tm], partial=False, layer_sizes=(200,200,200,200,200))
    k = 15
    scores = []
    print("Games starting...")
    for _ in range(k):
        game = mergedMain.mergeGame()
        moves = game.findLegalMoves()
        while len(moves) > 0:
            gameCopy = copy.deepcopy(game)
            move = agent.get_action(gameCopy, 1)
            game.playMove(move)
            moves = game.findLegalMoves()
        agent.terminate_learn(game)
        print(f"Score: {game.score}")
        scores.append(game.score)
    # Note: doesn't save fly training
    agent.save_model(filename="super_nnai.sav")
    avg_score = statistics.mean(scores)
    print(f"Avg match score: {avg_score}")

def draw_board(board):
    for row in board:
        for num in row:
            print(f" {num}", end='', flush=True)
        print()

# Observe nnai playing
def observe(model='nnai.sav'):
    agent = nnai.NNAI(filename=model)  # Epsilon is default, which is 0

    print("Model info:")
    print(f"Layers: {agent.model.n_layers_}, Outputs: {agent.model.out_activation_}")
    tm = os.listdir("training")
    tm.remove('.DS_Store')
    last = nnai.get_last_training()
    trainmulti=["training/" + x for x in tm if int(x.split("_")[1].split(".")[0])>=last-10]
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
        game_copy = copy.deepcopy(game)
        move = agent.get_action(game_copy, 1)
        game.playMove(move)
        
        draw_board(game.board)
        print("Piece: " + str(game.nextPiece))
        print("Score: " + str(game.score))
        print("Last move: " + str(move))
        print("Inference: " + str(agent.inference(game)))
        userin = input()
        if userin == '^[' or userin == '^C':
            return 0
        
        moves = game.findLegalMoves()
    print(f"Score: {game.score}")


if __name__ == '__main__':

    run_default(n=25, filename=None)

    observe()
    exit(0)

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