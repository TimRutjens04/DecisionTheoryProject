import os
import random
import sys
import time
import math
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from chess_logic.chess_5x5 import MiniChess
from models.minmax import MinimaxAI
from models.qlearning import QLearningAgent

# Training parameters
TOTAL_EPISODES = 100_000
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.1
EPSILON_DECAY = 0.9999

def is_improvement(prev_win_rate, new_win_rate, n_games=500):
    """Check if improvement is statistically significant"""
    if prev_win_rate is None:
        return True
    std_error = math.sqrt(prev_win_rate * (1 - prev_win_rate) / n_games)
    return new_win_rate > prev_win_rate + 2 * std_error  # 95% confidence

def evaluate_agent(agent, opponent_depth, num_games=500):
    """Evaluate agent performance over specified number of games"""
    results = {'wins': 0, 'losses': 0, 'draws': 0}
    minimax = MinimaxAI(depth=opponent_depth)
    
    for game_num in range(num_games):
        game = MiniChess()
        q_plays_white = game_num % 2 == 0
        
        while not game.is_game_over():
            if (game.turn == 'w') == q_plays_white:
                move = agent.choose_action(game)
            else:
                move = minimax.select_move(game)
            
            if move is None:
                break
            
            game.make_move(*move)
        
        winner = game.get_winner()
        if winner == 'draw':
            results['draws'] += 1
        elif (winner == 'w') == q_plays_white:
            results['wins'] += 1
        else:
            results['losses'] += 1
            
        if (game_num + 1) % 50 == 0:  # Progress update every 50 games
            print(f"Evaluation progress: {game_num + 1}/{num_games} games")
    
    win_rate = results['wins'] / num_games
    print(f"\nEvaluation Results (vs depth={opponent_depth}):")
    print(f"Wins: {results['wins']}, Losses: {results['losses']}, Draws: {results['draws']}")
    print(f"Win Rate: {win_rate:.2%}")
    return win_rate

def save_checkpoint(agent, episode, win_rate, metrics):
    """Save agent checkpoint and training metrics"""
    # Save agent
    checkpoint_dir = "saved_models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/checkpoint_ep{episode}_wr{win_rate:.2f}.pkl"
    agent.save(checkpoint_path)
    
    # Save metrics
    metrics_path = f"{checkpoint_dir}/training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    agent = QLearningAgent(
        name="Q-Learner",
        alpha=0.1,
        gamma=0.99,
        epsilon=INITIAL_EPSILON
    )
    
    metrics = {
        'episodes': [],
        'win_rates': [],
        'opponent_depths': [],
        'training_times': []
    }
    
    start_time = time.time()
    last_win_rate = None
    
    for episode in range(TOTAL_EPISODES):
        # Determine opponent depth and evaluation frequency
        if episode < 20_000:
            opponent_depth = 0
            eval_freq = 2_000
        elif episode < 50_000:
            opponent_depth = 1
            eval_freq = 3_000
        else:
            opponent_depth = 2
            eval_freq = 5_000
        
        # Training episode
        game = MiniChess()
        q_plays_white = random.random() < 0.5
        minimax = MinimaxAI(depth=opponent_depth)
        
        while not game.is_game_over():
            if (game.turn == 'w') == q_plays_white:
                action = agent.choose_action(game)
                if action is None:
                    break
                old_game = game.copy()
                game.make_move(*action)
                reward = agent.get_reward(old_game, action, game, 'w' if q_plays_white else 'b')
                agent.learn(old_game, action, reward, game.copy())
            else:
                move = minimax.select_move(game)
                if move:
                    game.make_move(*move)
        
        # Decay epsilon
        agent.epsilon = max(MIN_EPSILON, agent.epsilon * EPSILON_DECAY)
        
        # Evaluation and checkpointing
        if episode % eval_freq == 0:
            elapsed_time = time.time() - start_time
            print(f"\n{'='*50}")
            print(f"Episode {episode}/{TOTAL_EPISODES} ({episode/TOTAL_EPISODES:.1%})")
            print(f"Training time: {elapsed_time/3600:.1f} hours")
            print(f"Current ε: {agent.epsilon:.3f}")
            print(f"Opponent depth: {opponent_depth}")
            
            win_rate = evaluate_agent(agent, opponent_depth)
            
            # Record metrics
            metrics['episodes'].append(episode)
            metrics['win_rates'].append(win_rate)
            metrics['opponent_depths'].append(opponent_depth)
            metrics['training_times'].append(elapsed_time)
            
            # Save checkpoint
            save_checkpoint(agent, episode, win_rate, metrics)
            
            # Check for improvement and early stopping
            if episode > 30_000 and opponent_depth >= 1:
                if win_rate < 0.4 and not is_improvement(last_win_rate, win_rate):
                    print("\nPerformance plateaued - stopping training")
                    break
            
            last_win_rate = win_rate

if __name__ == "__main__":
    main()