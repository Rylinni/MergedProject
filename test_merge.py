import unittest
import mergedMain

class TestMerged(unittest.TestCase):
    def test_basicMerge(self):
        board = [[1,1,1,0,0],
                 [0,0,0,0,0], 
                 [0,0,0,0,0], 
                 [0,0,0,0,0],
                 [0,0,0,0,0]]
        move = ([1], [(0, 0)])
        game = mergedMain.mergeGame(board, move)
        resultBoard = [[2,0,0,0,0],
                       [0,0,0,0,0],
                       [0,0,0,0,0], 
                       [0,0,0,0,0],
                       [0,0,0,0,0]]
        score = 3
        self.assertEqual(game.score, score)
        self.assertEqual(game.board, resultBoard)
        
        board = [[0,1,1,1,0],
                 [0,2,2,2,0], 
                 [0,0,0,0,0], 
                 [0,0,0,0,0],
                 [0,0,0,0,0]]
        move = ([1,2], [(0, 3), (1, 3)])
        game = mergedMain.mergeGame(board, move)
        resultBoard = [[0,0,0,2,0],
                       [0,0,0,3,0],
                       [0,0,0,0,0], 
                       [0,0,0,0,0],
                       [0,0,0,0,0]]
        score = 6
        self.assertEqual(game.score, score)
        self.assertEqual(game.board, resultBoard)
        board = [[1,1,1,0,0],
                 [2,0,0,0,0], 
                 [0,0,0,0,0], 
                 [0,0,0,0,0],
                 [0,0,0,0,0]]
        move = ([1,2], [(0, 0), (1,0)])
        game = mergedMain.mergeGame(board, move)
        resultBoard = [[2,0,0,0,0],
                       [2,0,0,0,0],
                       [0,0,0,0,0], 
                       [0,0,0,0,0],
                       [0,0,0,0,0]]
        score = 3
        self.assertEqual(game.score, score)
        self.assertEqual(game.board, resultBoard)

    def test_doubleMerge(self):
        board = [[1,1,1,0,0],
                 [2,0,0,0,0], 
                 [2,0,0,0,0], 
                 [0,0,0,0,0],
                 [0,0,0,0,0]]
        move = ([1, 2], [(0, 0), (1, 0)])
        game = mergedMain.mergeGame(board, move)
        resultBoard = [[0,0,0,0,0],
                       [3,0,0,0,0],
                       [0,0,0,0,0], 
                       [0,0,0,0,0],
                       [0,0,0,0,0]]
        score = 18
        self.assertEqual(game.score, score)
        self.assertEqual(game.board, resultBoard)
        board = [[0,1,1,0,0],
                 [2,2,1,0,0], 
                 [0,3,3,0,0], 
                 [0,0,0,0,0],
                 [0,0,0,0,0]]
        move = ([1, 2], [(1, 2), (1, 1)])
        game = mergedMain.mergeGame(board, move)
        resultBoard = [[0,0,0,0,0],
                       [0,4,0,0,0],
                       [0,0,0,0,0], 
                       [0,0,0,0,0],
                       [0,0,0,0,0]]
        score = 108
        self.assertEqual(game.score, score)
        self.assertEqual(game.board, resultBoard)

    def test_megaMerge(self):
        board = [[0,2,2,0,0],
                 [3,3,2,4,4], 
                 [5,5,4,6,6], 
                 [0,7,7,0,0],
                 [0,0,0,0,0]]
        move = ([2, 4], [(1, 2), (2, 2)])
        game = mergedMain.mergeGame(board, move)
        resultBoard = [[0,0,0,0,0],
                       [0,0,0,0,0],
                       [0,0,0,0,0], 
                       [0,0,0,0,0],
                       [0,0,0,0,0]]
        score = 61200
        self.assertEqual(game.score, score)
        self.assertEqual(game.board, resultBoard)
if "__name__" == "__main__":
    unittest.main()