from chess_logic.chess_5x5 import MiniChess
import random

class MinimaxAI:
    def __init__(self, depth=2, name="minmax"):
        self.depth = depth
        self.name = name

    def evaluate(self, game):
        if game.winner == 'w':
            return float('inf')
        elif game.winner == 'b':
            return float('-inf')
        elif game.winner == 'draw':
            return 0

        score = 0
        piece_values = {'K': 1000, 'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1}
        for y in range(5):
            for x in range(5):
                piece = game.get_piece(x, y)
                if piece != '.':
                    value = piece_values.get(piece[1], 0)
                    score += value if piece[0] == 'w' else -value
        return score

    def minimax(self, game, depth, maximizing):
        if depth == 0 or game.is_game_over():
            return self.evaluate(game), None

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return self.evaluate(game), None

        best_move = None
        if maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                new_game = self.copy_game(game)
                new_game.make_move(*move)
                eval, _ = self.minimax(new_game, depth-1, False)
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in legal_moves:
                new_game = self.copy_game(game)
                new_game.make_move(*move)
                eval, _ = self.minimax(new_game, depth-1, True)
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
            return min_eval, best_move

    def copy_game(self, game):
        new_game = MiniChess()
        new_game.board = game.board.copy()
        new_game.turn = game.turn
        new_game.winner = game.winner
        new_game.halfmove_clock = game.halfmove_clock
        new_game.state_history = game.state_history.copy()
        return new_game

    def select_move(self, game):
        _, move = self.minimax(game, self.depth, game.turn == 'w')
        return move
