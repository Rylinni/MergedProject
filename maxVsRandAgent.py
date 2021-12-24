from typing import Optional, Tuple
from mergedMain import mergeGame
import math

class maxVsRand():
    def get_action(self, game_state: mergeGame, depth):
        _, action = self.maxMove(game_state, depth)
        return action
    
    def maxMove(self, state: mergeGame, depth) -> Tuple[float, Optional[Tuple]]:
        if depth <= 0:
            return state.score
        moves = state.findLegalMoves()
        if len(moves) == 0:
            return state.score
        else:
            bestVal = -math.inf
            bestMove = None
            for move in moves:
                val = self.averageMove(state.generateSuccessorBoard(move), depth-1)
                if val > bestVal:
                    bestVal = val
                    bestMove = move
        return (bestVal, bestMove)

    def averageMove(self, state: mergeGame, depth):
        pieces = state.allPossibleNextPieces()
        sum = 0
        for piece in pieces:
            val = self.maxMove(state.generateBoardNextPiece(piece), depth-1)
            if type(val) is tuple:
                val , _ = val
            sum += val
        return sum / len(pieces)

        
    