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
for match in range(100):

    game = mergedMain.mergeGame()
    moves = game.findLegalMoves()
    while len(moves) > 0:
        gameCopy = copy.deepcopy(game)
        move = agent.get_action(gameCopy, 2)
        game.playMove(move)
        moves = game.findLegalMoves()
    # print('game over')
    # drawBoard(game.board)
    print(f"match {match+1} score: {game.score}")
