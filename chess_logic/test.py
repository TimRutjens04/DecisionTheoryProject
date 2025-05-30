from chess_5x5 import *

game = MiniChess()
game.display()
print("Starting legal moves:", game.get_legal_moves())

move_success = game.make_move((1, 4), (0, 4))
print("\nMove success:", move_success)
game.display()

print("new legal moves:", game.get_legal_moves())