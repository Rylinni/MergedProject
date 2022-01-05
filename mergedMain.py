import random
import copy


class mergeGame:
    def __init__(self, board = None, move = None, score = 0, unlocks = None, piece = None):
        if board is None:
            board = [[0 for _ in range(5)] for _ in range(5)]
        
        if unlocks is None:
            unlocks = [1,2]
        else:
            unlocks = list(unlocks)

        if board and move:
            self.board = board
            self.score = score
            self.unlocks = unlocks
            self.lastAllPiecesUpdate = [1]
            self.evalBoard(move)
            
        elif board and score and unlocks and piece:
            self.board = board
            self.score = score
            self.unlocks = unlocks
            self.nextPiece = piece
            self.lastAllPiecesUpdate = [1]
        else:
            self.board = board
            self.score = score
            self.allPieces = []
            self.lastAllPiecesUpdate = [1]
            self.unlocks = unlocks
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

    def generateSuccessorBoard(self, move):
        # board = [[j for j in i] for i in self.board]
        board = copy.deepcopy(self.board)
        ghost = mergeGame(board=board, score=self.score, unlocks=self.unlocks)
        ghost.playMove(move)
        return ghost

    def generateBoardNextPiece(self, piece):
        ghost = mergeGame(self.board, score=self.score, unlocks=self.unlocks, piece=piece)
        return ghost

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

    def findLegalMoves(self, piece=None):
        if not piece:
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
                    
    def allPossibleNextPieces(self):
        usables = self.unlocks
        if usables == self.lastAllPiecesUpdate:
            return self.allPieces
        else:
            pieces = []
            for x in usables:
                for y in usables:
                    if x == y:
                        pieces.append([x])
                    else:
                        if not [y, x] in pieces:
                            pieces.append([x, y])
            self.allPieces = pieces
            return pieces

    def generateNextPiece(self):
        usables = self.unlocks
        random.shuffle(usables)
        nextPiece = [usables[0]]
        if random.randrange(2) == 1 and self.roomForDouble():
            nextPiece.append(usables[1])
        self.nextPiece = nextPiece

    def roomForDouble(self):
        bd = self.board
        for x in range(len(bd)-1)[1:]:
            for y in range(len(bd)-1)[1:]:
                if bd[x][y] == 0:
                    if bd[x-1][y] == 0:
                        return True
                    if bd[x+1][y] == 0:
                        return True
                    if bd[x][y-1] == 0:
                        return True
                    if bd[x][y+1] == 0:
                        return True
        return False

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
                try:
                    self.board[x-1][y-1] = 0
                except IndexError:
                    pass
            if right:
                try:
                    self.board[x-1][y+1] = 0
                except IndexError:
                    pass
            try:
                self.board[x-1][y] = 0
            except IndexError:
                pass
        if down:
            if left:
                try:
                    self.board[x+1][y-1] = 0
                except IndexError:
                    pass
            if right:
                try:
                    self.board[x+1][y+1] = 0
                except IndexError:
                    pass
            try:
                self.board[x+1][y] = 0
            except IndexError:
                pass
        if left:
            try:
                self.board[x][y-1] = 0
            except IndexError:
                pass
        if right:
            try:
                self.board[x][y+1] = 0
            except IndexError:
                pass
        self.board[x][y] = 0


    def upgrade(self, x, y):
        self.board[x][y] += 1
        self.merges += 1
        self.multiplier *= self.merges
        if self.board[x][y] == 8:
            self.explode(x, y)
        elif not (self.board[x][y] in self.unlocks):
            to_unlock = self.board[x][y]
            self.unlocks.append(to_unlock)
            self.allPossibleNextPieces()


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