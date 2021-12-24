import mergedMain
import nnai
import copy
import statistics
import os

# Generate test data, fit, repeat
def run_default(n=10, k=10):
    agent = nnai.NNAI(filename='nnai.sav', epsilon=.02)
    total_scores = []
    for match in range(n):
        print(f"Match {match+1}")
        scores = []
        try:
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
        except:
            print("There was an error.")
    avg_score = statistics.mean(total_scores)
    print(f"Avg total score: {avg_score}")

# Load in all test data generated from training directory,
# fit, and see what happens. Saves model to super_nnai.sav
def test_super():
    agent = nnai.NNAI()
    tm = os.listdir("training")
    print("Fitting...")
    agent.fit(trainmulti=["training/" + x for x in tm], partial=False, layer_sizes=(500,500,500))
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

run_default()
test_super()