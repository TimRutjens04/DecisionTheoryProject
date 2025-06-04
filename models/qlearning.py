import random
import pickle
import cloudpickle
from collections import defaultdict
from chess_logic.chess_5x5 import MiniChess

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.3, name="Q", q_table=None):
        self.q_table = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.name = name
        self.seen_states = set()

        if q_table is not None:
            print(f"[INFO] Loaded Q-table with {len(q_table)} entries.")
            self.q_table = defaultdict(float, q_table)
        else:
            print("[INFO] Initialized empty Q-table.")
            self.q_table = defaultdict(float)

    def get_state_key(self, game):
        return ''.join(''.join(row) for row in game.board) + game.turn

    def choose_action(self, game):
        state = self.get_state_key(game)
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return None

        # Add debugging information
        q_vals = [(self.q_table[(state, move)], move) for move in legal_moves]

        if random.random() < self.epsilon:
            chosen_move = random.choice(legal_moves)
            return chosen_move
        else:
            max_q = max(q_vals, key=lambda x: x[0])[0]
            best_moves = [move for q, move in q_vals if q == max_q]
            chosen_move = random.choice(best_moves)
            return chosen_move
        
    def learn(self, old_game, action, reward, new_game):
        old_state = self.get_state_key(old_game)
        new_state = self.get_state_key(new_game)
        legal_moves = new_game.get_legal_moves()

        if legal_moves:
            future_q = max([self.q_table[(new_state, move)] for move in legal_moves])
        else:
            future_q = 0

        old_q = self.q_table[(old_state, action)]
        self.q_table[(old_state, action)] += self.alpha * (reward + self.gamma * future_q - old_q)

    def save(self, filename='q_table.pkl'):
        with open(filename, 'wb') as f:
            cloudpickle.dump(dict(self.q_table), f)

    def load(self, filename='q_table.pkl'):
        with open(filename, 'rb') as f:
            self.q_table = defaultdict(float, pickle.load(f))

    def train(self, episodes=10000):
        for episode in range(episodes):
            game = MiniChess()
            self.seen_states.clear()  # Reset seen states for new episode
            agent_color = game.turn  # Remember agent's color
            
            while not game.is_game_over():
                state = self.get_state_key(game)
                action = self.choose_action(game)
                if action is None:
                    break
                    
                old_game = game.copy()
                game.make_move(*action)
                new_game = game.copy()
                
                # Calculate reward based on agent's perspective
                reward = self.get_reward(old_game, action, new_game, agent_color)
                self.learn(old_game, action, reward, new_game)

    def evaluate_position(self, game, perspective):
        """Evaluate board position from given color's perspective"""
        piece_values = {'K': 0, 'R': 5, 'B': 3}
        score = 0
        
        for y in range(5):
            for x in range(5):
                piece = game.get_piece(x, y)
                if piece != '.':
                    value = piece_values.get(piece[1], 0)
                    multiplier = 1.0
                    
                    # Bonus for controlling center
                    if 1 <= x <= 3 and 1 <= y <= 3:
                        multiplier += 0.2
                    
                    # Bonus for protecting king
                    if piece[1] == 'K':
                        friendly_pieces_nearby = 0
                        for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
                            if game.in_bounds(x+dx, y+dy):
                                nearby = game.get_piece(x+dx, y+dy)
                                if nearby != '.' and nearby[0] == piece[0]:
                                    friendly_pieces_nearby += 1
                        multiplier += friendly_pieces_nearby * 0.1
                    
                    final_value = value * multiplier
                    # Score relative to perspective
                    if piece[0] == perspective:
                        score += final_value
                    else:
                        score -= final_value
        
        return score

    def get_reward(self, old_game, action, new_game, agent_color):
        """Calculate reward based on action's effect"""
        # Check for illegal moves
        if not old_game.is_valid_move(*action):
            return -5.0
        
        # Get position evaluations from agent's perspective
        old_score = self.evaluate_position(old_game, agent_color)
        new_score = self.evaluate_position(new_game, agent_color)
        
        # Base reward is the improvement in position
        reward = new_score - old_score
        
        # Penalize repeated states
        new_state = self.get_state_key(new_game)
        if new_state in self.seen_states:
            reward -= 1.0
        self.seen_states.add(new_state)
        
        # Check for zero-effect moves
        if abs(reward) < 0.01:
            reward -= 1.0
        
        # Game ending rewards
        if new_game.winner:
            if new_game.winner == agent_color:
                reward += 100.0
            elif new_game.winner == 'draw':
                reward += 0
            else:
                reward -= 100.0
        
        return reward
    
    def write_to_file(self, text):
        with open(f"{self.name}_output.txt", "a") as f:
            f.write(text)