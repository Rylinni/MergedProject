# def drawBoard(board):
#     for row in board:
#         for num in row:
#             print(f" {num}", end='', flush=True)
#         print()
import random


class mergeGame:
    def __init__(self, board = None, move = None):
        if board and move:
            self.board = board
            self.score = 0
            self.evalBoard(move)
        else:
            self.board = [[0 for x in range(5)] for y in range(5)]
            self.score = 0
            self.unlocks = [1, 2]
            self.generateNextPiece()
    
    def playMove(self, move):
        values, coordinates = move
        if self.checkMoveViable(move):
            for i, (x, y) in enumerate(coordinates):
                self.board[x][y] = values[i]
            self.evalBoard(move)
            self.generateNextPiece()
        else:
            raise Exception("Invalid Move for current board")

    def checkMoveViable(self, move, insertMove = False):
        values, coordinates = move
        for coor in coordinates:
            if not(self.checkSpotEmpty(coor)):
                return False
        return True

    def checkSpotEmpty(self, coor):
        x, y = coor
        if x < 0 or x >= len(self.board) or y < 0 or y >= len(self.board):
            return False
        return self.board[x][y] == 0

    def findLegalMoves(self):
        piece = self.nextPiece
        # single piece
        viableMoves = []
        if len(piece) == 1:
            for i in range(len(self.board)):
                for j in range(len(self.board)):
                    move = (piece, [(i, j)])
                    if self.checkMoveViable(move):
                        viableMoves.append(move)
        else:
            # double piece
            for i in range(len(self.board)):
                for j in range(len(self.board)):
                    def buildAndCheck(coor1, coor2):
                        move = (piece, (coor1, coor2))
                        move2 = (piece, (coor2, coor1))
                        if self.checkMoveViable(move):
                            viableMoves.append(move)
                            viableMoves.append(move2)
                    # up
                    buildAndCheck((i, j), (i-1, j))
                    # down
                    buildAndCheck((i, j), (i+1, j))
                    # left
                    buildAndCheck((i, j), (i, j-1))
                    # right
                    buildAndCheck((i, j), (i, j+1))
        return viableMoves
                    


    def generateNextPiece(self):
        usables = self.unlocks
        random.shuffle(usables)
        nextPiece = [usables[0]]
        if random.randrange(2) == 1:
            nextPiece.append(usables[1])
        self.nextPiece = nextPiece

        

    def oneOfAKind(self, ls1, ls2):
        for x in ls2:
            if not x in ls1:
                ls1.append(x)
        return ls1
    
    def explode(self, x, y):
        up =  x > 0
        down = x+1 < len(self.board)
        left = y > 0
        right = y < len(self.board)
        if up:
            if left:
                self.board[x-1][y-1] = 0
            if right:
                self.board[x-1][y+1] = 0
            self.board[x-1][y] = 0
        if down:
            if left:
                self.board[x+1][y-1] = 0
            if right:
                self.board[x+1][y+1] = 0
            self.board[x+1][y] = 0
        if left:
            self.board[x][y-1] = 0
        if right:
            self.board[x][y+1] = 0
        self.board[x][y] = 0


    def upgrade(self, x, y):
        self.board[x][y] += 1
        self.merges += 1
        self.multiplier *= self.merges
        if self.board[x][y] == 8:
            self.explode(x, y)


    def getIdenticals(self, value, coor, identicals, partner = None):
        if partner:
            pval, (px, py) = partner
        else:
            pval = -1
            px, py = -1, -1
        x, y = coor
        board = self.board
        neighbors = []
        if x > 0 and not(px == x-1 and py == y):
            neighbors.append((x-1, y))
        if y > 0 and not(px == x and py == y-1):
            neighbors.append((x, y-1))
        if x < len(board) - 1 and not(px == x+1 and py == y):
            neighbors.append((x + 1, y))
        if y < len(board) - 1 and not(px == x and py == y+1):
            neighbors.append((x, y+1))
        for neighbor in neighbors:
            nx, ny = neighbor
            if value == board[nx][ny] and not neighbor in identicals  and pval != neighbors:
                identicals.append(neighbor)
                identicals = self.oneOfAKind(identicals, self.getIdenticals(value, neighbor, identicals, partner))
        return identicals

    
    def evalPiece(self, value, coor, partner = None, first = False):
        identicals = self.getIdenticals(value, coor, [coor], partner)
        if len(identicals) < 3:
            identicals = self.getIdenticals(value, coor, [coor])
        if len(identicals) >= 3 and not (partner and first and partner[0] == value):
            for neighbor in identicals:
                nx, ny = neighbor
                self.withheldPoints += self.board[nx][ny]
                if coor != neighbor:
                    self.board[nx][ny] = 0
            self.upgrade(coor[0], coor[1])
            return True
        return False
            

    def evalBoard(self, lastMove):
        values, coordinates = lastMove
        self.withheldPoints = 0
        self.multiplier = 1
        self.merges = 0


        if len(values) == 1:
            val = values[0]
            upgrade = self.evalPiece(val, coordinates[0])   
            while upgrade:
                val += 1
                upgrade = self.evalPiece(val, coordinates[0])
        else:
            val1 = values[0]
            val2 = values[1]
            if val1 < val2:
                smaller = (val1, coordinates[0])
                bigger = (val2, coordinates[1])
            else:
                smaller = (val2, coordinates[1])
                bigger = (val1, coordinates[0])
            biggerCanEval = 2 < len(self.getIdenticals(bigger[0], (bigger[1][0], bigger[1][1]), [(bigger[1][0], bigger[1][1])]))
            val = smaller[0]
            upgrade = self.evalPiece(val, smaller[1])
            while upgrade:
                val += 1
                upgrade = self.evalPiece(val, smaller[1], bigger, True)
            if biggerCanEval:
                self.withheldPoints = 0
                self.multiplier = 1
                self.merges = 0

            if self.board[bigger[1][0]][bigger[1][1]] == bigger[0]:
                val = bigger[0]
                upgrade = self.evalPiece(val, bigger[1], (smaller[0], smaller[1]))
                while upgrade:
                    val += 1
                    upgrade = self.evalPiece(val, bigger[1], (smaller[0], smaller[1]))
        self.score += self.multiplier * self.withheldPoints
                    


# board = [[1,1,1,0,0],
#                  [0,0,0,0,0], 
#                  [0,0,0,0,0], 
#                  [0,0,0,0,0],
#                  [0,0,0,0,0]]
# move = ([1], [(0, 0)])
game = mergeGame()
moves = game.findLegalMoves()
print('done')