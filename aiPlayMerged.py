import mergedMain
import maxVsRandAgent
import copy

def drawBoard(board):
    for row in board:
        for num in row:
            print(f" {num}", end='', flush=True)
        print()

def printOptions(moves):
    print()
    print("Options: ")
    for i, move in enumerate(moves):
        print(f" {i} :   {move}")
agent = maxVsRandAgent.maxVsRand()
game = mergedMain.mergeGame()
moves = game.findLegalMoves()
while len(moves) > 0:
    # drawBoard(game.board)
    # print(f"score: {game.score}")
    gameCopy = copy.deepcopy(game)
    move = agent.get_action(gameCopy, 8)
    game.playMove(move)
    moves = game.findLegalMoves()
print('game over')
drawBoard(game.board)
print(f"score: {game.score}")