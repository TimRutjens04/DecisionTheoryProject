import numpy as np
from collections import defaultdict
import copy

class MiniChess:
    def __init__(self):
        self.board = None
        self.turn = 'w'
        self.winner = None
        self.halfmove_clock = 0
        self.state_history = defaultdict(int)

        self.reset()


    SYMBOLS = {
        'bK': '♔', 'bQ': '♕', 'bR': '♖', 'bB': '♗', 'bN': '♘', 'bP': '♙',
        'wK': '♚', 'wQ': '♛', 'wR': '♜', 'wB': '♝', 'wN': '♞', 'wP': '♟',
        '.': '.'
    }

    def reset(self):
        self.board = np.array([
            ['.', 'bR', 'bK', 'bB', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.'],
            ['.', 'wR', 'wK', 'wB', '.']
        ])
        self.turn = 'w'
        self.winner = None
        self.halfmove_clock = 0
        self.state_history = defaultdict(int)
        self._record_state()

    def display(self):
        print("    0 1 2 3 4") # The random spaces are needed in the display (for columns)
        for y in range(5):
            row = [self.SYMBOLS.get(self.get_piece(x, y), '?') for x in range(5)]
            print(f"{y} | " + " ".join(row))
        print(f"Turn: {self.turn} | Halfmove clock: {self.halfmove_clock}")
        if self.winner:
            print(f"Game over! Winner: {self.winner}")
    
    def in_bounds(self, x, y):
        return 0 <= x < 5 and 0 <= y < 5
    
    def get_piece(self, x, y):
        return self.board[y][x]
    
    def set_piece(self, x, y, value):
        self.board[y][x] = value

    def is_game_over(self):
        return self.winner is not None
    
    def get_winner(self):
        return self.winner
    
    def _board_key(self):
        return str(self.board) + self.turn
    
    def copy(self):
        return copy.deepcopy(self)
    
    def _record_state(self):
        key = self._board_key()
        self.state_history[key] += 1
        if self.state_history[key] >= 3:
            self.winner = 'draw'
    
    def is_valid_move(self, from_pos, to_pos):
        fx, fy = from_pos
        tx, ty = to_pos

        if not (self.in_bounds(fx, fy) and self.in_bounds(tx, ty)):
            return False
        
        piece = self.get_piece(fx, fy)
        if piece == '.':
            return False
        
        color = piece[0]
        if color != self.turn:
            return False
        
        piece_type = piece[1]
        dx, dy = tx - fx, ty - fy

        target = self.get_piece(tx, ty)
        if target != '.' and target[0] == color:
            return False #Cant capture piece of its own

        match piece_type:
            case 'K':
                return max(abs(dx), abs(dy)) == 1
            case 'B':
                if abs(dx) != abs(dy):
                    return False #No diagonals
                
                step_x = np.sign(dx)
                step_y = np.sign(dy)
                x, y = fx + step_x, fy + step_y

                while (x, y) != (tx, ty):
                    if self.get_piece(x, y) != '.':
                        return False
                    x += step_x
                    y += step_y
                return True
            case 'R':
                if dx != 0 and dy != 0:
                    return False # A diagonal move
                
                step_x = np.sign(dx)
                step_y = np.sign(dy)
                x, y = fx + step_x, fy + step_y

                while (x, y) != (tx, ty):
                    if self.get_piece(x, y) != '.':
                        return False
                    x += step_x
                    y += step_y
                return True
        return False
    
    def is_in_check(self, color):
        king_pos = None
        for y in range(5):
            for x in range(5):
                if self.get_piece(x, y) == f"{color}K":
                    king_pos = (x, y)
                    break
            if king_pos:
                break
        if not king_pos:
            return True #Shouldnt happen but just to be safe
        
        opponent = 'b' if color == 'w' else 'w'
        for y in range(5):
            for x in range(5):
                piece = self.get_piece(x, y)
                if piece != '.' and piece[0] == opponent:
                    if self.is_valid_move((x, y), king_pos):
                        return True
        return False
    
    def make_move(self, from_pos, to_pos):
        if not self.is_valid_move(from_pos, to_pos):
            return False
        
        if self.winner:
            return False
        
        legal = False
        for move in self.get_legal_moves():
            if move == (from_pos, to_pos):
                legal = True
                break
        if not legal:
            return False
        
        fx, fy = from_pos
        tx, ty = to_pos

        moving_piece = self.get_piece(fx, fy)
        target_piece = self.get_piece(tx, ty)

        self.set_piece(tx, ty, moving_piece)
        self.set_piece(fx, fy, '.')

        if target_piece != '.':
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        if self.halfmove_clock >= 40:
            self.winner = 'draw'
            return True

        self.turn = 'b' if self.turn == 'w' else 'w'
        self._record_state()

        legal_moves = self.get_legal_moves()
        if not legal_moves:
            if self.is_in_check(self.turn):
                self.winner = 'w' if self.turn == 'b' else 'b'
            else:
                self.winner = 'draw' #No check but also no legal moves -> stalemate
        return True
    
    def get_legal_moves(self):
        moves = []
        for y in range(5):
            for x in range(5):
                piece = self.get_piece(x, y)
                if piece != '.' and piece[0] == self.turn:
                    for ty in range(5):
                        for tx in range(5):
                            if self.is_valid_move((x, y), (tx, ty)):
                                #Make temp move to check if own king in check
                                from_piece = self.get_piece(x, y)
                                to_piece = self.get_piece(tx, ty)
                                self.set_piece(tx, ty, from_piece)
                                self.set_piece(x, y, '.')
                                in_check = self.is_in_check(self.turn)
                                #Undo the temp move
                                self.set_piece(x, y, from_piece)
                                self.set_piece(tx, ty, to_piece)

                                if not in_check:
                                    moves.append(((x, y), (tx, ty)))
        return moves




    