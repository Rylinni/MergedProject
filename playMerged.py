import mergedMain

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

game = mergedMain.mergeGame()
moves = game.findLegalMoves()
while len(moves) > 0:
    drawBoard(game.board)
    print(f"score: {game.score}")
    printOptions(moves)
    while True:
        try:
            pick = int(input("Which move?"))
            move = moves[pick]
            break
        except:
            print("That's not a valid option!") 
    print()
    game.playMove(move)
    moves = game.findLegalMoves()
print('game over')
print(f"score: {game.score}")