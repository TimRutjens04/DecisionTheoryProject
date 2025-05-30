import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from chess_logic.chess_5x5 import MiniChess
from models.qlearning import QLearningAgent
from models.minmax import MinimaxAI
import pickle

# Load trained Q-table
with open("saved_models/q_table.pkl", "rb") as f:
    q_table = pickle.load(f)

q_agent = QLearningAgent(name="Q-Learner", q_table=q_table, epsilon=0.0)
minimax_agent = MinimaxAI(name="Minimax", depth=2)

NUM_MATCHES = 100
results = {'QAgent': 0, 'Minimax': 0, 'Draw': 0}

for i in range(NUM_MATCHES):
    game = MiniChess()

    q_plays_white = i % 2 == 0

    while not game.is_game_over():
        legal_moves = game.get_legal_moves()
        state = game._board_key()

        if (game.turn == 'w') == q_plays_white:
            move = q_agent.choose_action(game)
        else:
            move = minimax_agent.select_move(game)

        game.make_move(*move)

    winner = game.get_winner()
    if winner == 'w':
        results['QAgent'] += 1
    elif winner == 'b':
        results['Minimax'] += 1
    else:
        results['Draw'] += 1

    print(f"Game {i+1}: Winner - {winner}")

print("\nFinal Results After", NUM_MATCHES, "Matches:")
print(results)
