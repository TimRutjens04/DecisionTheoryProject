from chess_logic.chess_5x5 import MiniChess

class MinimaxAI:
    def __init__(self, depth=2, name="minmax"):
        self.depth = depth
        self.name = name
        self.color = None

    def evaluate(self, game):
        if self.color is None:
            self.color = game.turn

        if game.winner == 'w':
            return 100 if self.color == 'w' else -100
        elif game.winner == 'b':
            return 100 if self.color == 'b' else -100
        elif game.winner == 'draw':
            return 0

        score = 0
        piece_values = {'K': 0, 'Q': 9, 'R': 5, 'B': 3, 'N': 3, 'P': 1}
        for y in range(5):
            for x in range(5):
                piece = game.get_piece(x, y)
                if piece != '.':
                    if game.is_in_check(piece[0]):
                        score += -5 if piece[0] == self.color else 5
                    value = piece_values.get(piece[1], 0)
                    score += value if piece[0] == self.color else -value
        return score

    def minimax(self, game, depth, maximizing):
        if depth == 0 or game.is_game_over():
            eval_score = self.evaluate(game)
            # print(f"Leaf node evaluation: {eval_score} at depth {depth}")
            return eval_score, None

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            print("No legal moves found")
            return self.evaluate(game), None

        best_move = None
        if maximizing:
            max_eval = float('-inf')
            for move in legal_moves:
                new_game = self.copy_game(game)
                new_game.make_move(*move)
                eval, _ = self.minimax(new_game, depth-1, False)
                # print(f"Maximizing - Move: {move}, Eval: {eval}")
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
                # print(f"Minimizing - Move: {move}, Eval: {eval}")
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
            return min_eval, best_move

    def copy_game(self, game):
        new_game = MiniChess()
        new_game.board = [row.copy() for row in game.board]  # Deep copy for 2D array
        new_game.turn = game.turn
        new_game.winner = game.winner
        new_game.halfmove_clock = game.halfmove_clock
        new_game.state_history = game.state_history.copy()
        return new_game

    def select_move(self, game):
        # print(f"\nMinMax selecting move for {game.turn} at depth {self.depth}")
        # print(f"Current board:\n{game.board}")
        _, move = self.minimax(game, self.depth, game.turn == 'w')
        # print(f"Selected move {move} with evaluation {eval_score}")
        return move
