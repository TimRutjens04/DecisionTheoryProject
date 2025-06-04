import sys
import time
from pathlib import Path
import pygame

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from chess_logic.chess_5x5 import MiniChess
from models.minmax import MinimaxAI
from models.qlearning import QLearningAgent
from gui.gui import drawGrid, pygame, signal_game_end, draw_start_button

def simulate_match(agent1, agent2, delay=1.0):
    """Simulates a match between two agents with GUI visualization"""
    pygame.init()
    clock = pygame.time.Clock()
    game = MiniChess()
    running = True
    game_started = False
    game_ended = False
    winner = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            elif not game_started and event.type == pygame.MOUSEBUTTONDOWN:
                start_button = draw_start_button()
                if start_button.collidepoint(event.pos):
                    game_started = True
                    print("Match started!")

        if not game_started:
            drawGrid(game)
            draw_start_button()
            pygame.display.flip()
            clock.tick(60)
            continue

        if game_ended:
            # Keep showing final position and game end signal
            drawGrid(game)
            if winner:
                signal_game_end(winner)
            pygame.display.flip()
            clock.tick(60)
            continue

        if game.is_game_over():
            game_ended = True
            if game.winner:
                if game.winner == 'w':
                    winner = "White"
                    print(f"\nGame Over! Winner: {winner}")
                elif game.winner == 'b':
                    winner = "Black"
                    print(f"\nGame Over! Winner: {winner}")
                else:
                    winner = "Draw"
                    print(f"\nGame Over! The game was a draw.")
            else:
                winner = "Draw"
                print("\nGame Over! Draw")
            continue

        # Game logic for moves
        drawGrid(game)
        pygame.display.flip()

        current_agent = agent1 if game.turn == 'w' else agent2
        move = current_agent.select_move(game) if isinstance(current_agent, MinimaxAI) else current_agent.choose_action(game)
        
        if move:
            print(f"Player {game.turn} ({current_agent.name}) moves: {move}")
            game.make_move(*move)
            
            drawGrid(game)
            pygame.display.flip()
            time.sleep(delay)
        else:
            print(f"No valid moves for {current_agent.name}")
            game_ended = True

        clock.tick(60)

def main():
    minimax = MinimaxAI(depth=1, name="Minimax")
    qlearner = QLearningAgent(name="Q-Learner")
    
    try:
        qlearner.load("saved_models/best_model.pkl")
        print("Loaded trained Q-Learning model")
    except:
        print("No trained model found, using untrained Q-Learning agent")

    print("Starting match simulation...")
    simulate_match(qlearner, minimax, delay=1.5)

if __name__ == "__main__":
    main()