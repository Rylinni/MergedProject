import nnai
import os
import pickle

def construct_board(board_rep):
    new_board = [[]]
    i = 0
    x = 0
    for digit in board_rep:
        x += i * digit
        i += 1
        if i == 8:
            i = 0
            if len(new_board[len(new_board) - 1]) == 5:
                new_board.append([])
            new_board[len(new_board)-1].append(x)
            x = 0
    return new_board

def draw_board(board):
    for row in board:
        for num in row:
            print(f" {num}", end='', flush=True)
        print()

if __name__ == '__main__':
    tm = os.listdir("training")
    tm.remove('.DS_Store')
    last = nnai.get_last_training()

    nfiles = 5  # Read data from last x training files

    trainmulti=["training/" + x for x in tm if int(x.split("_")[1].split(".")[0])>=last-nfiles]
    training = [[], []]
    for fname in trainmulti:
        train = pickle.load(open(fname, 'rb'))
        training[0].extend(train[0])
        training[1].extend(train[1])

    # Flip through data
    x = ""
    i = 0
    while x != "q":
        draw_board(construct_board(training[0][i]))
        print(f"Actual score: {training[1][i]}")
        i += 1
        x = input()
